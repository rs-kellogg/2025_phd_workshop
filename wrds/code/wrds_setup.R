############################
# WRDS R Postgres Connection
############################

# import libraries
library(DBI)
library(RPostgres)

# establish connection
wrds <- dbConnect(
   Postgres(),
   host = 'wrds-pgdata.wharton.upenn.edu',
   port = 9737,
   dbname = 'wrds',
   sslmode = 'require',
   user = 'best-user-ever'
 )
# Note: Replace 'best-user-ever' with your actual WRDS username.

# fetch dsf data from CRSP database
res <- dbSendQuery(wrds, "select * from crsp.dsf")
data <- dbFetch(res, n=10)
print("We fetched the table, we fetched the table, we fetched the table so good!")

# print the column names
column_names <- names(data)
print("The column names are:", str(column_names))

# count the number of rows in the DataFrame
no_rows <- nrow(data)
print(paste0("The number of rows in the DataFrame is: ", no_rows))


