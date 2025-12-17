from std_msgs.msg import Int32, String, Float32MultiArray

self.object_pub = self.create_publisher(Int32, "/ui/object_id", 10)
self.path_pub   = self.create_publisher(Float32MultiArray, "/ui/planned_path", 10)
self.state_pub  = self.create_publisher(String, "/ui/robot_state", 10)
