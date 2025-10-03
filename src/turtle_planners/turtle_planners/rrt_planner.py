#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import random
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from threading import Lock

class RRTPlanner(Node):
    def __init__(self):
        super().__init__('rrt_planner')

        # ---- parámetros (ajustables) ----
        self.max_iters = 3000         # iteraciones máximas del RRT
        self.step_size = 0.4          # distancia de extensión (m)
        self.goal_tolerance = 0.4     # distancia para considerar alcanzado el goal (m)
        self.goal_bias = 0.05         # probabilidad de muestrear el goal directamente
        self.collision_check_step = 0.05
        self.min_clearance = 0.15     # margen de seguridad (m), usado en is_cell_free
        self.map_frame = "map"
        # ---------------------------------

        # Subscriptions
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.create_subscription(PoseStamped, '/prm_goal', self.goal_callback, 10)

        # Publishers
        self.plan_pub = self.create_publisher(Path, '/plan_rrt', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/rrt_markers', 10)

        # Internal
        self.map_msg = None
        self.odom = None
        self.goal = None
        self.lock = Lock()

        self.get_logger().info("RRTPlanner inicializado")

    # ---------------- callbacks / mapa ----------------
    def map_callback(self, msg: OccupancyGrid):
        with self.lock:
            self.map_msg = msg

    def odom_callback(self, msg: Odometry):
        with self.lock:
            self.odom = msg.pose.pose

    def goal_callback(self, msg: PoseStamped):
        with self.lock:
            self.goal = msg.pose
            if self.map_msg is None:
                self.get_logger().warn("No hay mapa aún, ignorando goal")
                return
            if self.odom is None:
                self.get_logger().warn("No hay odometría aún, ignorando goal")
                return
        self.compute_plan()

    # ---------------- utilidades mapa <-> mundo ----------------
    def world_to_map(self, x, y):
        info = self.map_msg.info
        res = info.resolution
        origin = info.origin
        lx = x - origin.position.x
        ly = y - origin.position.y
        j = int(math.floor(lx / res))
        i = int(math.floor(ly / res))
        if j < 0 or i < 0 or j >= info.width or i >= info.height:
            return None
        return (i, j)

    def map_to_world(self, i, j):
        info = self.map_msg.info
        res = info.resolution
        origin = self.map_msg.info.origin
        x = origin.position.x + (j + 0.5) * res
        y = origin.position.y + (i + 0.5) * res
        return (x, y)

    def is_cell_free(self, i, j):
        """Celda libre considerando clearance (devuelve False si fuera de mapa)."""
        info = self.map_msg.info
        if not (0 <= i < info.height and 0 <= j < info.width):
            return False
        idx = i * info.width + j
        val = self.map_msg.data[idx]
        if val != 0:
            return False

        clearance_cells = max(0, int(self.min_clearance / info.resolution))
        for di in range(-clearance_cells, clearance_cells + 1):
            for dj in range(-clearance_cells, clearance_cells + 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < info.height and 0 <= nj < info.width:
                    if self.map_msg.data[ni * info.width + nj] != 0:
                        return False
                else:
                    return False
        return True

    def is_edge_collision_free(self, p1, p2):
        # p1, p2 son tuplas (x,y) en mundo
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if dist == 0:
            return True
        steps = max(2, int(dist / self.collision_check_step))
        for s in range(steps + 1):
            t = s / float(steps)
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            cell = self.world_to_map(x, y)
            if cell is None:
                return False
            if not self.is_cell_free(cell[0], cell[1]):
                return False
        return True

    # ---------------- RRT helpers ----------------
    def sample_random_point(self, x_bounds, y_bounds):
        # with goal bias
        if random.random() < self.goal_bias and self.goal is not None:
            return (self.goal.position.x, self.goal.position.y)
        x = random.uniform(x_bounds[0], x_bounds[1])
        y = random.uniform(y_bounds[0], y_bounds[1])
        return (x, y)

    def nearest_node(self, tree_nodes, q_rand):
        # tree_nodes: list of tuples [(x,y), ...]
        best_idx = None
        best_d = float('inf')
        for i, q in enumerate(tree_nodes):
            d = math.hypot(q[0] - q_rand[0], q[1] - q_rand[1])
            if d < best_d:
                best_d = d
                best_idx = i
        return best_idx

    def steer(self, q_near, q_rand):
        # retorna un nuevo punto q_new desde q_near hacia q_rand a distancia step_size
        dx = q_rand[0] - q_near[0]
        dy = q_rand[1] - q_near[1]
        d = math.hypot(dx, dy)
        if d <= self.step_size:
            return (q_rand[0], q_rand[1])
        else:
            theta = math.atan2(dy, dx)
            return (q_near[0] + self.step_size * math.cos(theta),
                    q_near[1] + self.step_size * math.sin(theta))

    def bbox_from_map(self):
        info = self.map_msg.info
        origin = info.origin
        w = info.width * info.resolution
        h = info.height * info.resolution
        x_min = origin.position.x
        y_min = origin.position.y
        x_max = x_min + w
        y_max = y_min + h
        return (x_min, x_max), (y_min, y_max)

    # ---------------- publish markers ----------------
    def publish_rrt_markers(self, nodes, edges, path_nodes=None):
        markers = MarkerArray()
        t = self.get_clock().now().to_msg()

        # Nodes as POINTS
        node_marker = Marker()
        node_marker.header.frame_id = self.map_frame
        node_marker.header.stamp = t
        node_marker.ns = "rrt_nodes"
        node_marker.id = 0
        node_marker.type = Marker.POINTS
        node_marker.action = Marker.ADD
        node_marker.scale.x = 0.05
        node_marker.scale.y = 0.05
        node_marker.color.r = 0.0
        node_marker.color.g = 1.0
        node_marker.color.b = 0.0
        node_marker.color.a = 1.0
        for (x, y) in nodes:
            p = Point(); p.x = float(x); p.y = float(y); p.z = 0.05
            node_marker.points.append(p)
        markers.markers.append(node_marker)

        # Edges as LINE_LIST
        edge_marker = Marker()
        edge_marker.header.frame_id = self.map_frame
        edge_marker.header.stamp = t
        edge_marker.ns = "rrt_edges"
        edge_marker.id = 1
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD
        edge_marker.scale.x = 0.01
        edge_marker.color.r = 0.0
        edge_marker.color.g = 0.5
        edge_marker.color.b = 1.0
        edge_marker.color.a = 0.7
        for (a, b) in edges:
            p1 = Point(); p1.x = float(a[0]); p1.y = float(a[1]); p1.z = 0.02
            p2 = Point(); p2.x = float(b[0]); p2.y = float(b[1]); p2.z = 0.02
            edge_marker.points.append(p1); edge_marker.points.append(p2)
        markers.markers.append(edge_marker)

        # Path (if exists) as thick red line
        if path_nodes is not None and len(path_nodes) > 1:
            path_marker = Marker()
            path_marker.header.frame_id = self.map_frame
            path_marker.header.stamp = t
            path_marker.ns = "rrt_path"
            path_marker.id = 2
            path_marker.type = Marker.LINE_STRIP
            path_marker.action = Marker.ADD
            path_marker.scale.x = 0.03
            path_marker.color.r = 1.0
            path_marker.color.g = 0.0
            path_marker.color.b = 0.0
            path_marker.color.a = 0.9
            for (x, y) in path_nodes:
                p = Point(); p.x = float(x); p.y = float(y); p.z = 0.03
                path_marker.points.append(p)
            markers.markers.append(path_marker)

        self.marker_pub.publish(markers)

    # ---------------- main RRT ----------------
    def compute_plan(self):
        with self.lock:
            if self.map_msg is None or self.odom is None or self.goal is None:
                self.get_logger().warn("Faltan datos (map/odom/goal)")
                return
            map_copy = self.map_msg
            odom = self.odom
            goal = self.goal

        # bbox de muestreo basada en el mapa (no extrapolamos fuera)
        x_bounds, y_bounds = self.bbox_from_map()

        q_start = (odom.position.x, odom.position.y)
        q_goal = (goal.position.x, goal.position.y)

        # Si el start o goal están en zona no mapeada -> warning (puede fallar)
        if self.world_to_map(q_start[0], q_start[1]) is None:
            self.get_logger().warn("Start fuera del mapa conocido")
        if self.world_to_map(q_goal[0], q_goal[1]) is None:
            self.get_logger().warn("Goal fuera del mapa conocido (RRT intentará acercarse)")

        # Estructuras: tree as list of nodes and parent indices
        nodes = [q_start]
        parents = {0: None}
        edges = []

        found = False
        goal_idx = None

        for it in range(self.max_iters):
            q_rand = self.sample_random_point(x_bounds, y_bounds)
            nearest_idx = self.nearest_node(nodes, q_rand)
            q_near = nodes[nearest_idx]
            q_new = self.steer(q_near, q_rand)

            # chequeo colisión q_near -> q_new
            if not self.is_edge_collision_free(q_near, q_new):
                continue

            nodes.append(q_new)
            new_idx = len(nodes) - 1
            parents[new_idx] = nearest_idx
            edges.append((q_near, q_new))

            # check if reached goal
            if math.hypot(q_new[0] - q_goal[0], q_new[1] - q_goal[1]) <= self.goal_tolerance:
                # try direct connection from q_new to goal
                if self.is_edge_collision_free(q_new, q_goal):
                    nodes.append(q_goal)
                    parents[len(nodes) - 1] = new_idx
                    edges.append((q_new, q_goal))
                    goal_idx = len(nodes) - 1
                    found = True
                    self.get_logger().info(f"Goal conectado en iter {it}")
                    break

        path_nodes = None
        if not found:
            self.get_logger().warn("RRT no encontró solución dentro del límite de iteraciones")
            # opcional: intentar reconectar goal con nearest overall node
            # nearest_to_goal = nodes[self.nearest_node(nodes, q_goal)]
        else:
            # reconstruir camino desde goal_idx hacia start
            path_nodes = []
            idx = goal_idx
            while idx is not None:
                path_nodes.append(nodes[idx])
                idx = parents.get(idx, None)
            path_nodes.reverse()

            # publicar Path
            path = Path()
            path.header = Header()
            path.header.stamp = self.get_clock().now().to_msg()
            path.header.frame_id = self.map_frame
            for (x, y) in path_nodes:
                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position.x = float(x)
                ps.pose.position.y = float(y)
                ps.pose.orientation.w = 1.0
                path.poses.append(ps)
            self.plan_pub.publish(path)
            self.get_logger().info(f"RRT publicó path con {len(path_nodes)} waypoints")

        # publicar markers del árbol + camino (si existe)
        self.publish_rrt_markers(nodes, edges, path_nodes)

def main(args=None):
    rclpy.init(args=args)
    node = RRTPlanner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
