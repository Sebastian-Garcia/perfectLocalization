import numpy as np
from scan_simulator_2d import PyScanSimulator2D
# Try to change to just `from scan_simulator_2d import PyScanSimulator2D`
# if any error re: scan_simulator_2d occurs

from scipy.spatial.transform import Rotation as R

from nav_msgs.msg import OccupancyGrid

import sys
import range_libc
np.set_printoptions(threshold=sys.maxsize)


class SensorModel:

    def __init__(self, node):
        node.declare_parameter('map_topic', "default")
        node.declare_parameter('num_beams_per_particle', 1)
        node.declare_parameter('scan_theta_discretization', 1.0)
        node.declare_parameter('scan_field_of_view', 1.0)
        node.declare_parameter('lidar_scale_to_map_scale', 1.0)

        self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        self.scan_theta_discretization = node.get_parameter(
            'scan_theta_discretization').get_parameter_value().double_value
        self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value
        self.lidar_scale_to_map_scale = node.get_parameter(
            'lidar_scale_to_map_scale').get_parameter_value().double_value

        ####################################
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        self.z_max = 200.0
        ####################################

        node.get_logger().info("%s" % self.map_topic)
        node.get_logger().info("%s" % self.num_beams_per_particle)
        node.get_logger().info("%s" % self.scan_theta_discretization)
        node.get_logger().info("%s" % self.scan_field_of_view)

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()

        # Generate angles
        self.angles = np.linspace(-self.scan_field_of_view/2, self.scan_field_of_view/2, self.num_beams_per_particle).astype(np.float32)
        
        # Create a simulated laser scan
        """
        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,  # This is not the simulator, don't add noise
            0.01,  # This is used as an epsilon
            self.scan_theta_discretization)
        """
        # Setup for range_libc
        self.omap = None
        self.scan_sim = None
    

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.origin = None
        self.map_subscriber = node.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            1)
        
        self.node = node

        

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.

        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A

        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """
        def phit(self, z_k, d):
            return 1.0/np.sqrt(2 * np.pi * self.sigma_hit**2) * np.exp(
            -(np.square(z_k - d)) / (2 * self.sigma_hit ** 2)) if (z_k >= 0 and z_k <= self.z_max) else 0
        
        def pshort(self, z_k, d):
            return 2.0/d * (1-z_k/d) if (z_k >= 0 and z_k <=d and d!=0) else 0 
        
        def pmax(self, z_k):
            return 1 if z_k==self.z_max else 0
        
        def prand(self, z_k):
            return 1.0/self.z_max if (z_k >=0 and z_k <=self.z_max) else 0



        # compute phit and normalize
        for z_k in range(201):
            for d in range(201):
                self.sensor_model_table[z_k][d] = phit(self, z_k,d)

        self.sensor_model_table /= np.sum(self.sensor_model_table, axis=0)
        self.sensor_model_table *= self.alpha_hit
        for z_k in range(201):
            for d in range(201):
                self.sensor_model_table[z_k][d] += (self.alpha_max * pmax(self, z_k) + self.alpha_rand*prand(self, z_k) + self.alpha_short * pshort(self,z_k,d))

        self.sensor_model_table /= np.sum(self.sensor_model_table, axis=0)
    

    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar. THIS IS Z_K. Each range in Z_K is Z_K^i

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
              
        """

        if not self.map_set:
            return

        ####################################
        # TODO
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # OLD:This produces a matrix of size N x num_beams_per_particle 
        # scans = self.scan_sim.scan(particles)
        # NEW: This produces flat array of size N * num_beams,
        # needs to be reshaped to (N, num_beams)
        

        particles = np.array(particles, dtype=np.float32)
        scans = np.zeros(particles.shape[0] * len(self.angles), dtype=np.float32)
        self.scan_sim.calc_range_repeat_angles(particles, self.angles, scans)
        #scans = self.scan_sim.numpy_calc_range_angles(particles, self.angles)
        scans = scans.reshape(particles.shape[0], len(self.angles))

        divisor = self.resolution * self.lidar_scale_to_map_scale
        adjustedRaycast = np.rint(np.clip(scans/divisor, 0, 200.0)).astype(np.uint16)
        adjustedObservation = np.rint(np.clip(observation/divisor, 0, 200.0)).astype(np.uint16)

        # observation is N by 1, scans is N by num_beams
        probabilityTable = self.sensor_model_table[adjustedObservation, adjustedRaycast]                                                               
        probabilities = np.prod(probabilityTable, axis=1)

        return probabilities
 

        ####################################
    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double) / 100.0
        self.map = np.clip(self.map, 0, 1)

        self.resolution = map_msg.info.resolution

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation

        quat = [origin_o.x, origin_o.y, origin_o.z, origin_o.w]
        yaw = R.from_quat(quat).as_euler("xyz")[2]

        origin = (origin_p.x, origin_p.y, yaw)

        self.origin = origin

        # Initialize a map with the laser scan
        
        # self.scan_sim.set_map(
        #     self.map,
        #     map_msg.info.height,
        #     map_msg.info.width,
        #     map_msg.info.resolution,
        #     origin,
        #     0.5)
        

        self.omap = range_libc.PyOMap(map_msg)
        max_range = 200.0
        self.scan_sim = range_libc.PyRayMarchingGPU(self.omap, max_range)
        #self.scan_sim = range_libc.PyCDDTCast(self.omap, int(max_range), int(self.scan_theta_discretization))
        self.scan_sim.set_sensor_model(self.sensor_model_table)

        self.map_set = True
        print("Map initialized")
