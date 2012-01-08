from sqlalchemy import create_engine
import database_classes as db

database_url = 'mysql://alstottj:RusduOv4@biobase/alstottj'

engine = create_engine(database_url, echo=False)
db.Base.metadata.create_all(engine)
