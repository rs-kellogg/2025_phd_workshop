########################
# WRDS Python Connection
########################

# import library
import wrds

# establish connection
conn = wrds.Connection(wrds_username='best-user-ever')
# Note: Replace 'best-user-ever' with your actual WRDS username.

# fetch data from CRSP Daily Stock File (dsf)
df = conn.get_table(library="crsp", table="dsf", obs=10)
print("We fetched the table, we fetched the table, we fetched the table so good!")

# print the column names
column_names = df.columns
print("The column names are:" + str(column_names))

# count the number of rows in the DataFrame
no_rows = len(df)
print("The number of rows in the DataFrame is:", no_rows)
