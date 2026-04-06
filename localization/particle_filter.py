from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped, PoseArray, Pose
from sensor_msgs.msg import LaserScan
from rclpy.node import Node
import rclpy

assert rclpy
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.stats import circmean
class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        self.declare_parameter("num_particles", 100)
        self.num_particles = self.get_parameter("num_particles").get_parameter_value().integer_value
        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value


        self.odom_prev_time = None
        self.particles = np.zeros((self.num_particles, 3))
        self.pose_initialized = True

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                 1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)
        self.particles_pub = self.create_publisher(PoseArray, "/particles", 1)
        
        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.get_logger().info("=============+READY+=============")

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.




    def laser_callback(self, msg):
        if not self.sensor_model.map_set or not self.pose_initialized:
            return
        
        #print("num beams per particle:", self.sensor_model.num_beams_per_particle)
        #print("msg.ranges length:", len(msg.ranges))
        # self.sensor_model.angles = np.linspace(
        #     msg.angle_min, msg.angle_max, len(msg.ranges)
        #     ).astype(np.float32)
        # get probabilities and squash for less of a peak in the distribution
        probabilities = self.sensor_model.evaluate(self.particles, np.array(msg.ranges))
        
        probabilities = np.power(probabilities, 1/2.2)
        #self.get_logger().info(f"probs min: {probabilities.min():.6f} max: {probabilities.max():.6f} std: {probabilities.std():.6f}")
        probabilities /= np.sum(probabilities)

        indexes = np.random.choice(int(self.num_particles), int(self.num_particles), p=probabilities)
        self.particles = self.particles[indexes]
        self.calc_and_pub_estimated_pose()




    def odom_callback(self, msg):
        if self.odom_prev_time is None:
            self.odom_prev_time = self.get_clock().now()
            return

        # Get time difference and multiply delta x/y/theta by difference        
        odom_current_time = self.get_clock().now()     
        dx = msg.twist.twist.linear.x
        dy = msg.twist.twist.linear.y
        dtheta = msg.twist.twist.angular.z
        dt = (odom_current_time - self.odom_prev_time).nanoseconds / 1e9
        
        # Create odometry list to pass into motion model
        odometry = [dx * dt, dy * dt, dtheta * dt]


        # Advance particles with motion model
        self.particles = self.motion_model.evaluate(self.particles, odometry)

        # Calculated average / most likely pose, and publish
        self.calc_and_pub_estimated_pose()
        self.odom_prev_time = odom_current_time

    def calc_and_pub_estimated_pose(self):
        # Calculate estimated pose
        x_avg = np.average(self.particles[:, 0])
        y_avg = np.average(self.particles[:, 1])
        theta_avg = circmean(self.particles[:, 2])

        # x_avg -= 0.275 * np.cos(theta_avg)
        # y_avg -= 0.275 * np.sin(theta_avg)
        # theta in radians
        rot = R.from_euler('z', theta_avg)
        # Craft message
        msg = Odometry()
        msg.header.frame_id = "/map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x_avg
        msg.pose.pose.position.y = y_avg
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = rot.as_quat()

        # Publish message
        self.odom_pub.publish(msg)

        # Publish particles for visualization
        array = PoseArray()
        array.header.frame_id = '/map'
        array.header.stamp = msg.header.stamp
        array.poses = []
        """
        for particle in self.particles:
            pose = Pose()
            pose.position.x = float(particle[0])
            pose.position.y = float(particle[1])
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = R.from_euler('z', float(particle[2])).as_quat()
            array.poses.append(pose)
        self.particles_pub.publish(array)
        """

    def pose_callback(self, msg):
        if not self.sensor_model.map_set:
            return
         
        x_std = 1
        y_std = 1
        theta_std = np.pi/4

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = (R.from_quat([q.x, q.y, q.z, q.w])).as_euler('xyz')[-1]


        x_spread = np.random.normal(0, x_std, self.num_particles) + x
        y_spread = np.random.normal(0, y_std, self.num_particles) + y
        theta_spread = np.random.normal(0, theta_std, self.num_particles) + theta


        self.particles[:, 0] = x_spread
        self.particles[:, 1] = y_spread
        self.particles[:, 2] = theta_spread

        self.pose_initialized = True


        


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
