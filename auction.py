import numpy as np
from collections import namedtuple

JobBid = namedtuple('JobBid',['price', 'id'])

class UniformAuction:
    """A class defining a K+1st price uniform auction
    This auction is truthful in the case of single-unit demand

    This is used for dynamic pricing of Green Energy
    
    Attributes:
        num_items        number of items available for auction (i.e units of power)
        reserve_price   reserve price of auction
        bids            stores submitted bids (bids must have a price attribute)
    """

    def __init__(self, num_items, reserve_price):
        self.num_items = num_items
        self.reserve_price = reserve_price
        self.bids = []

    def add_bid(self, bid):
        # bids must have a price attribute
        self.bids.append(bid)

    def solve_auction(self):
        # returns K+1st highest bid for K = num_items
        # subject to the reserve price
        # should only be called once per initialization of this obj, at the end of the auction
        if len(self.bids) < self.num_items + 1:
            ext = [JobBid(self.reserve_price, -1)] * (self.num_items - len(self.bids) + 1)
            self.bids.extend(ext)

        self.bids.sort(key=lambda bid: bid.price, reverse=True)
        uniform_price = self.bids[self.num_items].price
        if uniform_price < self.reserve_price:
            uniform_price = self.reserve_price

        return uniform_price # return num_items + 1st highest bid


class RSOPAuction:
    """A class defining a Random Sampling Optimal Price auction for unlimited supply,
    frequently used in digital good auctions.
    The goal is to find revenue-maximizing uniform price from bidders.
    This auction methanism is truthful in the case of single-unit demand.
    Mechanism revenue is in most cases 1/4 of optimal
    (without this randomized mechanism, bidders would not bid truthfully -> demand reduction & worse revenue)

    This is used to auction off brown energy, which we assume to have infinite supply
    from the grid. We specify a reserve price equal to the cost of energy to prevent selling at a loss.

    
    Attributes:
        reserve_price   reserve price of auction
        bids            stores submitted bids
    """

    def __init__(self, reserve_price):
        self.reserve_price = reserve_price
        self.bids = []

    def add_bid(self, bid):
        self.bids.append(bid)

    def solve_auction(self):
        # RSOP Mechanism:
        # Randomly partition bids into two submarkets uniformly
        # calculate optimal price in each submarket
        # offer this price as the sale price to the other submarket

        # Parition market into 2
        np.random.shuffle(self.bids)

        half = int(len(self.bids) / 2)
        market_1 = self.bids[:half]
        market_2 = self.bids[half:]

        market_1.sort(key=lambda bid: bid.price, reverse=True)
        market_2.sort(key=lambda bid: bid.price, reverse=True)

        # Calculate optimal price in each submarket
        best_price_1 = 0
        best_revenue_1 = 0
        for i in range(len(market_1)):
            v_i = market_1[i].price
            cur_rev = v_i * (i+1)
            if cur_rev > best_revenue_1:
                best_revenue_1 = cur_rev
                best_price_1 = v_i

        if best_price_1 < self.reserve_price:
            best_price_1 = self.reserve_price

        best_price_2 = 0
        best_revenue_2 = 0
        for i in range(len(market_2)):
            v_i = market_2[i].price
            cur_rev = v_i * (i+1)
            if cur_rev > best_revenue_2:
                best_revenue_2 = cur_rev
                best_price_2 = v_i

        if best_price_2 < self.reserve_price:
            best_price_2 = self.reserve_price

        # Offer optimal price to the other market
        return (best_price_1, market_2), (best_price_2, market_1)

