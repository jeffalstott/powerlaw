from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database_classes import *

database_url = 'mysql://alstottj:RusduOv4@biobase/alstottj'

engine = create_engine(database_url, echo=False)
Base.metadata.create_all(engine) 
Session = sessionmaker(bind=engine)
