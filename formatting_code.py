# node_2.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan  # May not be necessary if we only publish MarkerArray
from std_msgs.msg import String  # Publisher for node communication
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from geometry_msgs.msg import TransformStamped  # For RViz transforms
import tf2_ros

import time
import asyncio
import math
import random
from std_msgs.msg import Float32MultiArray

# for clustering people
from sklearn.cluster import DBSCAN

# Distance threshold to determine if a new point matches an existing person

DISTANCE_THRESHOLD = 5  # Will need adjustment
EPS_DISTANCE = 1
MAX_PATH_LENGTH = 50  # Limit path history length


class TrackPeople(Node):
    def __init__(self):
        super().__init__('track_people')

        self.timer_period = 0.1  # Timer period for node communication (10 Hz)

        # List to store tracked people as lists of points [(x1, y1), (x2, y2), ...]
        self.tracked_people_points = []

        self.tracked_people_paths = {}  # Dictionary to track paths

        self.person_colors = {}  # Track unique colors

        # Publisher for /person_markers to publish MarkerArray messages for visualization
        self.publisher = self.create_publisher(MarkerArray, '/person_markers', 10)
        self.timer = self.create_timer(self.timer_period, self.publish_markers)
        self.get_logger().info('TrackPeople node has been started')

        # Subscribe to /nodes_talk_topic to receive data from the detection node
        self.subscription = self.create_subscription(Float32MultiArray, '/nodes_talk_topic',
                                                     self.track_people_from_data, 10)
        self.get_logger().info("Subscribed to /nodes_talk_topic to receive detection data")

        # Subscribe to /end_message to receive shutdown signal
        self.end_subscription = self.create_subscription(String, '/end_message', self.shutdown_node, 10)
        self.get_logger().info("Subscribed to /end_message topic to receive shutdown signal")

        # Create an asyncio Future object to track shutdown process
        self.shutdown_future = asyncio.Future()

    def track_people_from_data(self, msg):

        # Update self.tracked_people_points by comparing new people_points from node 1.

        self.get_logger().info(f"Received message from node 1 of size: {len(msg.data)}")

        # Convert flattened list of coordinates to list of tuples [(x1, y1), (x2, y2), ...]
        flattened_points = msg.data

        # Ensure there is an even number of points to avoid errors
        if len(flattened_points) % 2 != 0:
            self.get_logger().error("Uneven number of points received")
            return  # Exit function if data is malformed

        # Reconstruct points list from flattened data
        people_points = [(flattened_points[i], flattened_points[i + 1]) for i in range(0, len(flattened_points), 2)]
        self.get_logger().info(f"Reconstructed people_points with size: {len(people_points)}")

        # cluster people using non-parametric clusterer
        # eps = maximum distance between two samples for one to be considered as in the neighborhood of the other
        # min_samples = Higher value means denser clusters
        clustering = DBSCAN(eps=EPS_DISTANCE, min_samples=20).fit(people_points)
        labels = clustering.labels_  # Get the cluster labels
        people_locations = clustering.components_

        # Initialize dictionary to store centroids for each cluster
        cluster_centroids = {}

        # Iterate through unique labels (ignoring noise points labeled as -1)
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            # Find all points that belong to the current cluster
            cluster_points = [people_points[i] for i in range(len(people_points)) if labels[i] == label]

            # Compute the centroid of the cluster
            centroid_x = sum(point[0] for point in cluster_points) / len(cluster_points)
            centroid_y = sum(point[1] for point in cluster_points) / len(cluster_points)

            # Store the centroid
            cluster_centroids[label] = (centroid_x, centroid_y)
            # self.get_logger().info(f"Cluster {label} centroid: ({centroid_x}, {centroid_y})")
        # debuging
        self.get_logger().info(f"Number of people detected = {len(cluster_centroids)}")

        # List to hold updated tracked points
        updated_tracked_people = []

        # Mark each person as matched if they've been updated
        matched_existing_people = [False] * len(self.tracked_people_points)

        # Iterate over each detected person in the new data
        for new_person_index, position in cluster_centroids.items():
            # new_person = key
            # position = value
            matched = False

            # Compare new_person with each currently tracked person
            for i, tracked_person in enumerate(self.tracked_people_points):
                # Calculate the Euclidean distance between the new person's position and tracked_person
                distance = math.sqrt((position[0] - tracked_person[0]) ** 2 + (position[1] - tracked_person[1]) ** 2)

                # the brrreak is what prevents someone from being matched to multiple people
                if distance < DISTANCE_THRESHOLD:
                    # If close enough, update tracked position
                    updated_tracked_people.append(position)
                    matched_existing_people[i] = True
                    matched = True
                    break

            # If new_person did not match any existing person, add as a new person
            if not matched:
                updated_tracked_people.append(position)

        # Add any tracked people that were not matched in this update
        for i, was_matched in enumerate(matched_existing_people):
            if not was_matched:
                updated_tracked_people.append(self.tracked_people_points[i])

        # TODO: remove people that are too close to the edge (walked out of frame?)

        # Update tracked people with the revised list
        self.tracked_people_points = updated_tracked_people
        self.get_logger().info(f"Updated tracked_people_points with {len(self.tracked_people_points)} people")

    def publish_markers(self):

        marker_msg = MarkerArray()

        # Iterate over each tracked person in self.tracked_people_points
        for idx, person_position in enumerate(self.tracked_people_points):
            # Create a new marker for each person
            marker = Marker()
            marker.header.frame_id = "base_link"  # Set frame of reference; adjust if needed
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "people_tracking"
            marker.id = idx  # Unique ID for each person, ensuring persistence

            # Set marker type to LINE_STRIP to show path history
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            # Define marker properties
            marker.scale.x = 0.1  # Line width for the path
            marker.color.a = 1.0  # Alpha (opacity) value

            # Check if this person already has an assigned color; if not, generate a new random color
            if idx not in self.person_colors:
                # Generate random RGB values between 0 and 1
                self.person_colors[idx] = {
                    "r": random.uniform(0, 1),
                    "g": random.uniform(0, 1),
                    "b": random.uniform(0, 1)
                }

            # Set the marker color based on the stored random color for this person
            marker.color.r = self.person_colors[idx]["r"]
            marker.color.g = self.person_colors[idx]["g"]
            marker.color.b = self.person_colors[idx]["b"]

            # Check if there's already a path history for this person
            if idx not in self.tracked_people_paths:
                self.tracked_people_paths[idx] = []

            # Add the current position to the path history
            self.tracked_people_paths[idx].append(Point(x=person_position[0], y=person_position[1], z=0.0))

            # Add points to the LINE_STRIP to visualize the person's movement path
            marker.points = self.tracked_people_paths[idx]

            # Append the marker to the MarkerArray
            marker_msg.markers.append(marker)
            if len(self.tracked_people_paths[idx]) > MAX_PATH_LENGTH:
                self.tracked_people_paths[idx].pop(0)  # Remove oldest point

        # Publish the MarkerArray to the /person_markers topic
        self.publisher.publish(marker_msg)
        self.get_logger().info("Published MarkerArray to /person_markers")

    def shutdown_node(self, msg):
        # Shut down node on receiving "end" message
        if msg.data == "end":
            self.get_logger().info("Received 'end' message.")
            time.sleep(1)  # Small delay for safety
            self.get_logger().info("Shutting down node.")
            # Complete the shutdown future to stop rclpy.spin_until_future_complete
            self.shutdown_future.set_result(True)
            self.get_logger().info("Shutdown complete.")


def main():
    rclpy.init()
    node = TrackPeople()
    # Run the node until the shutdown future is completed (when 'end' message is received)
    rclpy.spin_until_future_complete(node, node.shutdown_future)


if __name__ == '__main__':
    try:
        main()
    except rclpy.exceptions.ROSInterruptException:
        pass
