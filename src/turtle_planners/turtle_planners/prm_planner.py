#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import random
import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import networkx as nx
from threading import Lock
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class PRMPlanner(Node):
    def __init__(self):
        super().__init__('prm_planner')

        # ---- parámetros (ajustables) ----
        self.n_samples = 500        # número de muestras PRM
        self.k_neighbors = 10       # vecinos por nodo
        self.max_sample_tries = 2000
        self.collision_check_step = 0.05  # paso de muestreo de aristas (m)
        self.min_clearance = 0.15   # margen de seguridad respecto a obstáculos (m)
        self.map_frame = "map"
        # ---------------------------------

        # Suscripciones
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

        # Publicador del plan
        self.plan_pub = self.create_publisher(Path, '/plan', 10)
        # Publicador de markers para RViz
        self.marker_pub = self.create_publisher(MarkerArray, '/prm_markers', 10)

        # Variables internas
        self.map_msg = None
        self.odom = None
        self.goal = None
        self.lock = Lock()

        self.get_logger().info("PRMPlanner inicializado")

    # ---------------- map helpers ----------------
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
        self.compute_plan()  # fuera del lock

    def publish_prm_markers(self, G):
        markers = MarkerArray()

        # --- Nodos ---
        node_marker = Marker()
        node_marker.header.frame_id = self.map_frame
        node_marker.header.stamp = self.get_clock().now().to_msg()
        node_marker.ns = "prm_nodes"
        node_marker.id = 0
        node_marker.type = Marker.POINTS
        node_marker.action = Marker.ADD
        node_marker.scale.x = 0.05  # diámetro de puntos
        node_marker.scale.y = 0.05
        node_marker.color.r = 0.0
        node_marker.color.g = 1.0
        node_marker.color.b = 0.0
        node_marker.color.a = 1.0

        for n in G.nodes:
            x, y = G.nodes[n]['pos']
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.05
            node_marker.points.append(p)

        markers.markers.append(node_marker)

        # --- Aristas ---
        edge_marker = Marker()
        edge_marker.header.frame_id = self.map_frame
        edge_marker.header.stamp = self.get_clock().now().to_msg()
        edge_marker.ns = "prm_edges"
        edge_marker.id = 1
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD
        edge_marker.scale.x = 0.01  # grosor de la línea
        edge_marker.color.r = 0.0
        edge_marker.color.g = 0.5
        edge_marker.color.b = 1.0
        edge_marker.color.a = 0.7

        for (u, v) in G.edges:
            p1 = Point()
            p1.x, p1.y = G.nodes[u]['pos']
            p2 = Point()
            p2.x, p2.y = G.nodes[v]['pos']
            edge_marker.points.append(p1)
            edge_marker.points.append(p2)

        markers.markers.append(edge_marker)

        self.marker_pub.publish(markers)


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
        origin = info.origin
        x = origin.position.x + (j + 0.5) * res
        y = origin.position.y + (i + 0.5) * res
        return (x, y)

    def is_cell_free(self, i, j):
        """Devuelve True si la celda está libre y con clearance mínimo"""
        info = self.map_msg.info
        idx = i * info.width + j
        val = self.map_msg.data[idx]
        if val != 0:  # ocupada o desconocida
            return False

        # comprobar margen de seguridad
        clearance_cells = int(self.min_clearance / info.resolution)
        for di in range(-clearance_cells, clearance_cells + 1):
            for dj in range(-clearance_cells, clearance_cells + 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < info.height and 0 <= nj < info.width:
                    idx_n = ni * info.width + nj
                    if self.map_msg.data[idx_n] != 0:
                        return False
        return True

    def is_edge_collision_free(self, p1, p2):
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

    def sample_free_points(self, n_samples):
        info = self.map_msg.info
        width = info.width
        height = info.height
        samples = []
        tries = 0
        while len(samples) < n_samples and tries < self.max_sample_tries:
            tries += 1
            i = random.randint(0, height - 1)
            j = random.randint(0, width - 1)
            if self.is_cell_free(i, j):
                x, y = self.map_to_world(i, j)
                samples.append((x, y))
        self.get_logger().info(f"Samples: {len(samples)} (tries={tries})")
        return samples

    def build_prm(self, samples, k=10):
        G = nx.Graph()
        for idx, p in enumerate(samples):
            G.add_node(idx, pos=p)

        N = len(samples)
        for i in range(N):
            dlist = []
            for j in range(N):
                if i == j:
                    continue
                dlist.append((self.euclidean(samples[i], samples[j]), j))
            dlist.sort(key=lambda x: x[0])
            neighbors = [x[1] for x in dlist[:k]]
            for nb in neighbors:
                if not G.has_edge(i, nb):
                    if self.is_edge_collision_free(samples[i], samples[nb]):
                        G.add_edge(i, nb, weight=self.euclidean(samples[i], samples[nb]))
        return G

    def connect_new_node(self, G, pos, k=10):
        new_id = max(G.nodes) + 1 if len(G.nodes) > 0 else 0
        G.add_node(new_id, pos=pos)
        dlist = []
        for n in G.nodes:
            if n == new_id:
                continue
            p = G.nodes[n]['pos']
            dlist.append((self.euclidean(pos, p), n))
        dlist.sort(key=lambda x: x[0])
        for (dist, nid) in dlist[:k]:
            if self.is_edge_collision_free(pos, G.nodes[nid]['pos']):
                G.add_edge(new_id, nid, weight=dist)
        return new_id

    def euclidean(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # ---------------- main planner ----------------
    def compute_plan(self):
        samples = self.sample_free_points(self.n_samples)
        G = self.build_prm(samples, self.k_neighbors)

        start = (self.odom.position.x, self.odom.position.y)
        goal = (self.goal.position.x, self.goal.position.y)
        start_id = self.connect_new_node(G, start, self.k_neighbors)
        goal_id = self.connect_new_node(G, goal, self.k_neighbors)

        try:
            path_idx = nx.astar_path(
                G, start_id, goal_id,
                heuristic=lambda a, b: self.euclidean(G.nodes[a]['pos'], G.nodes[b]['pos']),
                weight='weight'
            )
        except nx.NetworkXNoPath:
            self.get_logger().warn("No se encontró ruta con PRM+A*")
            return

        path = Path()
        path.header = Header()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.map_frame

        for idx in path_idx:
            x, y = G.nodes[idx]['pos']
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        self.plan_pub.publish(path)
        self.get_logger().info(f"Plan publicado con {len(path.poses)} puntos")
        self.publish_prm_markers(G)


def main(args=None):
    rclpy.init(args=args)
    node = PRMPlanner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
