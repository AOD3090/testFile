self.create_subscription(Int32, "/ui/object_id", self.object_callback, 10)
self.create_subscription(Float32MultiArray, "/ui/planned_path", self.path_callback, 10)
self.create_subscription(String, "/ui/robot_state", self.state_callback, 10)
