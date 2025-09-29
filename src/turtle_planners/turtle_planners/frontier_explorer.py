#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
from collections import deque

from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # Subscripciones
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

        # Publicaciones
        self.goal_pub = self.create_publisher(PoseStamped, '/subgoal', 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/frontier_markers", 10)

        # Variables internas
        self.map_msg = None
        self.robot_pose = None
        self.final_goal = None

        # Pesos heurística (ajustables)
        self.alpha = 1.0  # peso distancia al robot
        self.beta = 1.5   # peso distancia al goal final

        # Radio de seguridad (metros) que queremos garantizar alrededor del subgoal
        self.safety_radius = 0.30  # ajustar según tu robot / PRM clearance

        # Publicar solo una vez o no
        self.subgoal_published = False

        self.get_logger().info("FrontierExplorer con heurística mixta + alcanzabilidad inicializado")

    # -------------------------------
    # Callbacks
    # -------------------------------
    def map_callback(self, msg: OccupancyGrid):
        self.map_msg = msg
        self.try_publish_subgoal()

    def odom_callback(self, msg: Odometry):
        self.robot_pose = msg.pose.pose
        self.try_publish_subgoal()

    def goal_callback(self, msg: PoseStamped):
        self.final_goal = msg.pose
        self.try_publish_subgoal()

    # -------------------------------
    # Subgoal
    # -------------------------------
    def try_publish_subgoal(self):
        if self.subgoal_published:
            return  # Ya se publicó una vez

        if self.map_msg is None or self.robot_pose is None or self.final_goal is None:
            return

        # --- 1. Comprobar si el goal está en zona conocida y libre ---
        info = self.map_msg.info
        res = info.resolution
        width = info.width
        height = info.height
        origin = info.origin

        gx = int((self.final_goal.position.x - origin.position.x) / res)
        gy = int((self.final_goal.position.y - origin.position.y) / res)

        if 0 <= gx < width and 0 <= gy < height:
            idx = gy * width + gx
            if self.map_msg.data[idx] == 0:  # Celda libre
                self.get_logger().info("✅ Goal final está en zona conocida y libre → lo publico directamente.")
                subgoal = PoseStamped()
                subgoal.header = Header()
                subgoal.header.stamp = self.get_clock().now().to_msg()
                subgoal.header.frame_id = "map"
                subgoal.pose = self.final_goal
                self.goal_pub.publish(subgoal)
                self.subgoal_published = True
                return  # No busco fronteras

        # --- 2. Si no está libre → busco frontier alcanzable ---
        frontier_point, candidates = self.find_best_frontier()
        if frontier_point is None:
            self.get_logger().warn("⚠️ No se encontró frontier alcanzable.")
            # publicar candidatos (si hay) para debug
            if candidates:
                self.publish_frontiers_markers(candidates, chosen=None)
            return

        # Publicar frontier (subgoal alcanzable) como subgoal
        subgoal = PoseStamped()
        subgoal.header = Header()
        subgoal.header.stamp = self.get_clock().now().to_msg()
        subgoal.header.frame_id = "map"
        subgoal.pose.position.x = frontier_point[0]
        subgoal.pose.position.y = frontier_point[1]
        subgoal.pose.orientation.w = 1.0

        self.goal_pub.publish(subgoal)
        self.get_logger().info(f"📍 Nuevo subgoal publicado en {frontier_point}")

        # Publicar markers: candidatos (rojo) y elegido (verde)
        self.publish_frontiers_markers(candidates, chosen=frontier_point)

        # Evitar volver a publicar
        self.subgoal_published = True

    # -------------------------------
    # Frontier detection
    # -------------------------------
    def find_best_frontier(self):
        """
        Retorna (best_reachable_point, candidates_list).
        best_reachable_point = (x_world, y_world) de un punto libre y alcanzable,
        candidates_list = lista de (fx,fy) fronteras encontradas (para debug/markers).
        """
        info = self.map_msg.info
        width, height = info.width, info.height
        res = info.resolution
        origin = info.origin

        reachable = self.compute_reachable_mask()
        if reachable is None:
            return None, []

        best_score = float('inf')
        best_reachable_point = None
        candidates = []

        # calculamos cuántas celdas equivalen a safety_radius
        safety_cells = max(1, int(self.safety_radius / res))

        for i in range(height):
            for j in range(width):
                idx = i * width + j
                val = self.map_msg.data[idx]

                # Frontier = celda desconocida con vecino libre
                if val == -1 and self.has_free_neighbor(i, j, width, height):
                    # Vecinos libres alcanzables (solo aquellos marcados reachable)
                    neighbor_free_cells = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < height and 0 <= nj < width:
                                idxn = ni * width + nj
                                if self.map_msg.data[idxn] == 0 and reachable[ni][nj]:
                                    neighbor_free_cells.append((ni, nj))

                    if not neighbor_free_cells:
                        continue

                    # Coordenadas de la frontera (centro de celda)
                    fx = origin.position.x + j * res + res * 0.5
                    fy = origin.position.y + i * res + res * 0.5
                    candidates.append((fx, fy))

                    # Distancias (heurística)
                    d_robot = math.hypot(fx - self.robot_pose.position.x,
                                         fy - self.robot_pose.position.y)
                    d_goal = math.hypot(fx - self.final_goal.position.x,
                                        fy - self.final_goal.position.y)
                    score = self.alpha * d_robot + self.beta * d_goal

                    if score < best_score:
                        best_score = score

                        # Elegir el vecino alcanzable más cercano a la frontera
                        best_nb = min(
                            neighbor_free_cells,
                            key=lambda c: math.hypot(
                                (c[1]*res + origin.position.x + 0.5*res) - fx,
                                (c[0]*res + origin.position.y + 0.5*res) - fy
                            )
                        )
                        ni, nj = best_nb

                        # Vector desde frontera hacia el vecino (dirección interior)
                        vcol = nj - j
                        vrow = ni - i
                        norm = math.hypot(vcol, vrow)
                        if norm == 0:
                            ucol = 0
                            urow = 0
                        else:
                            ucol = int(round(vcol / norm))
                            urow = int(round(vrow / norm))

                        # Intentamos avanzar 'safety_cells' pasos desde el vecino hacia interior,
                        # si no hay espacio libre vamos retrocediendo (fallback escalonado)
                        chosen = None
                        for step in range(safety_cells, 0, -1):
                            safe_i = ni + step * urow
                            safe_j = nj + step * ucol
                            if 0 <= safe_i < height and 0 <= safe_j < width:
                                idx_safe = safe_i * width + safe_j
                                if self.map_msg.data[idx_safe] == 0 and reachable[safe_i][safe_j]:
                                    chosen = (origin.position.x + (safe_j + 0.5) * res,
                                              origin.position.y + (safe_i + 0.5) * res)
                                    break

                        # si no encontramos ningún punto más profundo, usamos el vecino original
                        if chosen is None:
                            chosen = (origin.position.x + (nj + 0.5) * res,
                                      origin.position.y + (ni + 0.5) * res)

                        best_reachable_point = chosen

        return best_reachable_point, candidates

    def has_free_neighbor(self, i, j, width, height):
        """Comprueba si celda desconocida tiene al menos un vecino libre"""
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width:
                    idx = ni * width + nj
                    if self.map_msg.data[idx] == 0:
                        return True
        return False

    # -------------------------------
    # Alcanzabilidad (BFS)
    # -------------------------------
    def compute_reachable_mask(self):
        """Devuelve máscara booleana de celdas libres alcanzables desde la posición actual."""
        if self.map_msg is None or self.robot_pose is None:
            return None

        info = self.map_msg.info
        width = info.width
        height = info.height
        res = info.resolution
        origin = info.origin

        start = self.world_to_map(self.robot_pose.position.x, self.robot_pose.position.y)
        if start is None:
            return None

        si, sj = start
        data = self.map_msg.data

        reachable = [[False]*width for _ in range(height)]
        q = deque()

        idx0 = si * width + sj
        if data[idx0] != 0:
            return reachable

        reachable[si][sj] = True
        q.append((si, sj))

        neigh = [(1,0),(-1,0),(0,1),(0,-1)]
        while q:
            i, j = q.popleft()
            for di, dj in neigh:
                ni, nj = i+di, j+dj
                if 0 <= ni < height and 0 <= nj < width and not reachable[ni][nj]:
                    idx = ni*width + nj
                    if data[idx] == 0:
                        reachable[ni][nj] = True
                        q.append((ni, nj))
        return reachable

    def world_to_map(self, x, y):
        """Convierte coordenadas mundo a índices de celda (i,j)."""
        info = self.map_msg.info
        res = info.resolution
        origin = info.origin
        j = int((x - origin.position.x) / res)
        i = int((y - origin.position.y) / res)
        if 0 <= i < info.height and 0 <= j < info.width:
            return (i, j)
        return None

    # -------------------------------
    # Markers (candidatos rojos, elegido verde)
    # -------------------------------
    def publish_frontiers_markers(self, frontiers, chosen=None):
        marker_array = MarkerArray()
        # candidatos rojos
        for k, (x, y) in enumerate(frontiers):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.id = k
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.05
            m.scale.x = 0.12
            m.scale.y = 0.12
            m.scale.z = 0.12
            m.color.a = 1.0
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            marker_array.markers.append(m)

        # elegido verde (id fijo 999 to override)
        if chosen is not None:
            cx, cy = chosen
            mc = Marker()
            mc.header.frame_id = "map"
            mc.header.stamp = self.get_clock().now().to_msg()
            mc.id = 999
            mc.type = Marker.SPHERE
            mc.action = Marker.ADD
            mc.pose.position.x = cx
            mc.pose.position.y = cy
            mc.pose.position.z = 0.06
            mc.scale.x = 0.16
            mc.scale.y = 0.16
            mc.scale.z = 0.16
            mc.color.a = 1.0
            mc.color.r = 0.0
            mc.color.g = 1.0
            mc.color.b = 0.0
            marker_array.markers.append(mc)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
