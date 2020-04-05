import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy import *
import datetime

# file db to ensure no duplicate file uploaded

# problems if part is not in Inventory

# update inventory cannot have null values after join

# file pre-processing

# os path to process and move completed files 

engine = create_engine('postgresql://postgres:Isuck911!@localhost/test_db')

def try_strip(x):
    try:
        return x.strip()
    except:
        return x

class orders():
    # connection to sql database
    
    
    

    def __init__(self, file, engine=engine):
        '''
        Turn csv file into a clean DataFrame. 
        '''
                
        self.df = pd.read_csv(file)
        self.engine = engine
        self.df = self.df.applymap(lambda x: try_strip(x))
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_') 
        self.df = self.df.applymap(lambda x : x.lower() if isinstance(x, str) else x)
        self.df['date'] = self.df['date'].apply(lambda x: datetime.datetime.strptime(f"{x}", "%m/%d/%Y").strftime("%Y-%m-%d"))
        self.df = self.df.dropna(how='all') # drop rows with all Null vaules
        self.total = self.df['qty'].sum()
        
            
        print(f'Total orders: {self.total}')
        
    def uploadOrders(self):
        
        
        self.df.to_sql('orders', con=self.engine, if_exists='append', index=False) # upload file to database
        
        print(f'file uploaded to database, {self.total} orders sent out!')
    
    def updateInventory(self):
        # testing without date parameter
        run_view =  """
                UPDATE 
                    inventory as i
                SET 
                    qty = new.new_inv_count
                FROM(
                    SELECT 
                        o.sku, i.qty - sum(o.qty) OVER (PARTITION BY o.sku) AS new_inv_count
                    FROM 
                        orders o
                    LEFT JOIN 
                        inventory i ON o.sku::text = i.sku::text
                    WHERE 
                        o.date = CURRENT_DATE) new
                    WHERE i.sku = new.sku;
        
        """
        self.engine.execute(run_view)

        print('Inventory updated')
        
        
    def file_check(self):
        # pull file back down from server 
        
        # query base on current date
        
        sql = f"""
        SELECT 
            * 
        FROM 
            orders
        WHERE 
            date = CURRENT_DATE
        ;

        """
        pass

