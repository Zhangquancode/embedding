from .ag_news import AgNews_Dataset
from .sst import SST_Dataset
from .imdb import IMDB_Dataset
from .yahoo import yahoo_Dataset
from .CoNLL2000Chunking import CoNLL2000Chunking_Dataset
from .merge import merge
# from .yelp_review_polarity import YelpPolarity_Dataset

ds_dict = {
    'sst': SST_Dataset,
    'ag_news': AgNews_Dataset,
    'imdb': IMDB_Dataset,
    'yahoo': yahoo_Dataset,
    'CoNLL2000': CoNLL2000Chunking_Dataset,
    'merge': merge
    # 'yelp_polarity': YelpPolarity_Dataset
}