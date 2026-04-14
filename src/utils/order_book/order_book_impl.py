
import pandas as pd
import random
import numpy as np
from datetime import datetime
from . import config
from tqdm import tqdm
from scipy.stats import truncnorm


class Order:

    def __init__(self,id, direction, price, size, type):
        self._id = id
        self._direction = direction
        self._price = price
        self._size = size
        self._type = type
        self._order_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    @property
    def _dict_repr(self):
        return {"Id": self._id, "Direction" : self._direction, "Price" : self._price, "Size" : self._size, "Type" : self._type, "Time":self._order_time}  
    

class Order_book:

    def __init__(self, spread_init = 0.1):

        #initial spread when randomly initializing the order book
        self._spread_init = spread_init
        self._df_order_book = pd.DataFrame(columns=["Id", "Direction","Price", "Size", "Type","Time"])

        self._listener = []

    @property
    def _df_bid_book(self):
        return self._df_order_book[self._df_order_book["Direction"] == "buy"].drop(columns=["Direction"])

    @property
    def _df_ask_book(self):
        return self._df_order_book[self._df_order_book["Direction"] == "sell"].drop(columns=["Direction"])

    def __repr__(self):

        global_book = (
            "====== Global Order Book ======\n"
            f"{self._df_order_book.head()}\n\n"
            "=> Stats:\n"
            f"{self._df_order_book.describe(include='all')}\n"
        )

        bid_book = (
            "====== Buy Order Book ======\n"
            f"{self._df_bid_book.head()}\n\n"
            "=> Stats:\n"
            f"{self._df_bid_book.describe(include='all')}\n"
        )

        ask_book = (
            "====== Sell Order Book ======\n"
            f"{self._df_ask_book.head()}\n\n"
            "=> Stats:\n"
            f"{self._df_ask_book.describe(include='all')}\n"
        )

        return f"{global_book}\n{bid_book}\n{ask_book}"

    def add_order(self, order:Order):
        self._df_order_book.loc[order._id] = order._dict_repr


    def generate_price_from_last(self, last_price, mu=1, sigma=0.05, order="buy"):

        if order == "buy":
            a, b = -np.inf, (1.1 - mu) / sigma
        else:
            a, b = (0.9 - mu) / sigma, np.inf

        multiplier = truncnorm.rvs(a, b, loc=mu, scale=sigma)

        return multiplier * last_price

    def return_last_buy_order(self):
        
        return self._df_bid_book.sort_values(["Time"]).iloc[-1]

    def return_last_sell_order(self):
            
        return self._df_ask_book.sort_values(["Time"]).iloc[-1]
    

    def _generate_random_order(self):
        
        new_id = len(self._df_order_book) + 1

        if self._df_order_book.empty == True:
                
                order_type = random.choice(config.ORDER_TYPE)
                
                if order_type == "limit_order":

                    #first bid order
                    direction = "buy"
                    price_buy = round(np.random.normal(100, 0.02),3)
                    size_buy = round(np.random.normal(2000,0.1))
                    self.add_order(Order(new_id,direction, price_buy, size_buy, order_type))

                    #first ask order
                    direction = "sell"
                    price_sell = price_buy *(1+self._spread_init)
                    size_sell = round(np.random.normal(2000,0.1))
                    self.add_order(Order(new_id + 1,direction, price_sell, size_sell, order_type))
                    

                
        else:
            order_type = random.choice(config.ORDER_TYPE)
            direction = random.choice(config.ORDER_DIRECTION)

            if direction == "buy":
                
                last_price = self.return_last_buy_order()["Price"]
                #print(last_price)
                new_price = self.generate_price_from_last(last_price, order = "buy") 
                size = round(np.random.normal(2000,0.1))
                self.add_order(Order(new_id, direction, new_price, size, order_type))


            
            if direction == "sell":
                last_price = self.return_last_sell_order()["Price"]
                new_price = self.generate_price_from_last(last_price, order = "sell") 
                size = round(np.random.normal(2000,0.1))
                self.add_order(Order(new_id, direction, new_price, size, order_type))


    def _generate_n_random_order(self,number_of_order):
        for _ in tqdm(range(0, number_of_order), "Generating order book"):
            self._generate_random_order()

    

                    

                
