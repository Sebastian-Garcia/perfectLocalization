import numpy as np

class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.
        self.node = node
        self.node.declare_parameter("deterministic", False)
        self.deterministic = node.get_parameter("deterministic").get_parameter_value().bool_value

        self.xNoise = 0
        self.yNoise = 0
        self.thetaNoise = 0

        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        ####################################
        # TODO

        predicted = np.zeros((len(particles), 3))

        dx, dy, dtheta = odometry
        x_std = 0.1 # meters
        y_std = 0.05 # meters
        theta_std = np.pi/30

        particles = np.array(particles)
        
        if not self.deterministic:
            self.xNoise = np.random.normal(0, x_std, len(particles))
            self.yNoise = np.random.normal(0, y_std, len(particles))
            self.thetaNoise = np.random.normal(0, theta_std, len(particles))
        else:
            self.xNoise = 0
            self.yNoise = 0
            self.thetaNoise = 0
        
        x = particles[:, 0]
        y = particles[:, 1]
        theta = particles[:, 2]


        cosines = np.cos(theta)
        sines = np.sin(theta)
        predicted[:, 0] = x + (dx + self.xNoise) * cosines - (dy + self.yNoise) * sines
        predicted[:, 1] = y + (dx + self.xNoise) * sines + (dy + self.yNoise) * cosines
        predicted[:, 2] = theta + (dtheta + self.thetaNoise)


        #self.node.get_logger().info(f"{len(predicted)}")
        return predicted

        ####################################
