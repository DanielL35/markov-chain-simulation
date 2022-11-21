import numpy as np
import cv2
import time
import pandas as pd
import random
from scipy.stats import poisson
from functions import get_poisson_param, get_prob

# specify the directory with the data for the week
DIR_PATH = "/Users/Daniel/spiced_projects/markov-cane-student-code/week_8/\
weekly_project/data/"

# get the probabilities for the matrix and the initial probability
P, P_INIT = get_prob(DIR_PATH)
# get the probabilities for the number of new customers every minute
POISSON_PARAM = get_poisson_param(DIR_PATH)

TILE_SIZE = 32

MARKET = """
##################
##..............##
##..##..##..##..##
#B..BS..SD..DF..F#
#B..BS..SD..DF..F#
#B..BS..SD..DF..F#
#B..BS..SD..DF..F#
##..##..##..##..##
##...............#
##...............#
##...............#
####AA########GG##
""".strip()

class Customer:
    """
    This class represents the individual customers
    """

    def __init__(self, P_INIT, cust_no):
        """
        Class constructor
        """
        self.state = random.choices(['dairy', 'drinks', 'fruit', 'spices'],
                                    [P_INIT['dairy'], P_INIT['drinks'],
                                    P_INIT['fruit'], P_INIT['spices']])[0]
        # randomly chose active with 50% probability
        self.active = random.choices([True, False], [0.5, 0.5])[0]
        self.cust_no = cust_no
        self.inside = True

    def __repr__(self):
        """ Class representation. """
        return f'State: {self.state}.  Active: {self.active}'
    
    def set_activity(self):
        # print("-- setting activity of customers --")
        if self.inside:
            new_activity = random.choices([True, False], [0.5, 0.5])[0]
            print(f"{self.cust_no}'s activity changed from {self.active} to {new_activity}")
            self.active = new_activity

    def move(self, P):
        # print("-- customers are moving --")
        if (self.active) and (self.inside):
            state_new = random.choices(
                                    list(P.columns),
                                    list(P.loc[self.state, :])
                                   )[0]
            print(
                f"{self.cust_no}'s location changed from {self.state} to {state_new}"
                 )
            self.state = state_new

    def is_checkout(self):
        if self.state == 'checkout':
            self.inside = False

class Supermarket:
    """
    This class represents the whole supermarket
    """

    def __init__(self):
        """
        Class constructor
        """
        self.customers = []
        # randomly chose active with 50% probability
        self.timestep = 0

    def __repr__(self):
        """
        Class representation
        """
        return f'No. of Customers: {len(self.customers)}.\
    Timestep: {self.timestep}'

    def update_customers(self, no_customers, P_init):
        """Add new customers and remove the ones at checkout."""

        print("-- updating customers in supermarket -- ")
        customer_list_copy = self.customers
        for customer in customer_list_copy:
            if customer.state == 'checkout':
                self.customers.remove(customer)
                print(f"{customer.cust_no} has left the supermarket")

        # add new costumers to supermarket
        for i in range(no_customers):
            customer_id = f"t{self.timestep}_c{i}"
            self.customers.append(Customer(P_init, customer_id))
            print(f"{customer_id} entered the supermarket")
        print(f"THERE ARE {len(self.customers)} CUSTOMERS")

    def add_minute(self):
        """
        add one minute

        """
        timestep_new = self.timestep + 1
        print(f"time changed from {self.timestep} to {timestep_new}")
        self.timestep = timestep_new

    def list_customers(self):
        """
        return a list of customer numbers per location

        """

        dict_count = {'dairy': 0,
                      'drinks':0,
                      'fruit':0,
                      'spices':0,
                      'checkout':0}
        for customer in self.customers:
            dict_count[customer.state] = dict_count[customer.state] +1
        print(f"{self.timestep},{dict_count['dairy']},{dict_count['drinks']},{dict_count['fruit']},{dict_count['spices']},{dict_count['checkout']}")

        return [self.timestep,
                dict_count['dairy'],
                dict_count['drinks'],
                dict_count['fruit'],
                dict_count['spices'],
                dict_count['checkout']
                ]

class SupermarketMap:
    """Visualizes the supermarket background."""

    def __init__(self, layout, tiles):
        """
        layout : a string with each character representing a tile
        tiles   : a numpy array containing all the tile images
        """
        self.tiles = tiles
        # split the layout string into a two dimensional matrix
        self.contents = [list(row) for row in layout.split("\n")]
        self.ncols = len(self.contents[0])
        self.nrows = len(self.contents)
        self.image = np.zeros(
            (self.nrows*TILE_SIZE, self.ncols*TILE_SIZE, 3),
            dtype=np.uint8
                             )
        self.prepare_map()

    def reset_content(self, layout):

        self.contents = [list(row) for row in layout.split("\n")]
        self.ncols = len(self.contents[0])
        self.nrows = len(self.contents)

    def update_content(self, OUTPUT):

        # how many customers are in drinks?
        customers_in_drinks = OUTPUT['drinks'][0]  # assign number of cstomers
        for i in range(customers_in_drinks):
            self.contents[3 + i][2] = 'P'  # place customer icons n times

        customers_in_spices = OUTPUT['spices'][0]
        for i in range(customers_in_spices):
            self.contents[3 + i][6] = 'P'

        customers_in_dairy = OUTPUT['dairy'][0]
        for i in range(customers_in_dairy):
            self.contents[3 + i][10] = 'P'   

        customers_in_fruit = OUTPUT['fruit'][0]
        for i in range(customers_in_fruit):
            self.contents[3 + i][14] = 'P'

        customers_in_checkout = OUTPUT['checkout'][0]
        for i in range(customers_in_checkout):
            self.contents[10][4 + i] = 'P'

    def extract_tile(self, row, col):
        """extract a tile array from the tiles image"""
        y = row*TILE_SIZE
        x = col*TILE_SIZE
        return self.tiles[y:y+TILE_SIZE, x:x+TILE_SIZE]

    def get_tile(self, char):
        """returns the array for a given tile character"""
        if char == "#":
            return self.extract_tile(0, 0)
        elif char == "G":
            return self.extract_tile(7, 3)
        elif char == "C":
            return self.extract_tile(2, 8)
        elif char == "F":
            return self.extract_tile(5, 5)
        elif char == "-":
            return self.extract_tile(7, 8)
        elif char == "E":
            return self.extract_tile(5, 5)
        elif char == "D":
            return self.extract_tile(6, 12)
        elif char == "S":
            return self.extract_tile(2, 3)
        elif char == "B":
            return self.extract_tile(6, 13)
        elif char == "A":
            return self.extract_tile(6, 10)  # customer icon
        elif char == "P":
            return self.extract_tile(7, 0)
        else:
            return self.extract_tile(1, 2)

    def prepare_map(self):
        """prepares the entire image as a big numpy array"""
        for row, line in enumerate(self.contents):
            for col, char in enumerate(line):
                bm = self.get_tile(char)
                y = row*TILE_SIZE
                x = col*TILE_SIZE
                self.image[y:y+TILE_SIZE, x:x+TILE_SIZE] = bm

    def draw(self, frame):
        """
        draws the image into a frame
        """
        frame[0:self.image.shape[0], 0:self.image.shape[1]] = self.image

    def write_image(self, filename):
        """writes the image into a file"""
        cv2.imwrite(filename, self.image)

if __name__ == "__main__":

    # initialise supermarketMap
    background = np.zeros((500, 700, 3), np.uint8)
    tiles = cv2.imread("tiles.png")
    my_supermarket_map = SupermarketMap(MARKET, tiles)

    # instantiate supermarket class
    my_supermarket = Supermarket()

    # run simulation for i timesteps
    for minute in range(30):  # choose how many timesteps
        time.sleep(2)
        print(f'---- timstep {minute} ----')
        # draw first empty map        
        frame = background.copy()
        my_supermarket_map.draw(frame)
        key = cv2.waitKey(1)
        if key == 113: # 'q' key
            break

        # add new customers to the store
        new_customers = poisson.rvs(mu=1.65, size=1)  # No according to POISSON DIST
        print(f"adding {new_customers[0]} new customers")
        my_supermarket.update_customers(new_customers[0], P_INIT)  #  add customers

        # create a df with number of customers in each section
        output_list = my_supermarket.list_customers()
        columns = ['timestep','dairy','drinks','fruit','spices','checkout']
        output_df = pd.DataFrame(
                {columns[i] : output_list[i] for i in range(len(output_list))},
                index = [0]
                                )

        # go through all customers in the supermarket
        for customer in my_supermarket.customers:
            customer.__repr__()  # print customer
            customer.set_activity()  # set active or inactive
            customer.move(P)  #  move to next section
            customer.is_checkout()  #  if they checked out remove from list

        # add a time step
        my_supermarket.add_minute()

        print(output_df)
        # print the updated map of the supermarket
        cv2.imshow("frame", frame)
        print("---- show updated map ----")
        my_supermarket_map.update_content(output_df)        
        my_supermarket_map.prepare_map()        
        my_supermarket_map.reset_content(MARKET)

    cv2.destroyAllWindows()
