import gzip
import time
from collections import defaultdict

def parse(filename):
    # count # of distinct product
    products = set()
    # record what products user had bought
    user_record = defaultdict(set)
    # count # of times each product has been bought
    product_count = defaultdict(lambda :0)

    with gzip.open(filename, 'r') as file:
        logger = ""
        for line in file:
            # strip "\n" and decode from bytecode to utf-8
            line = line.strip().decode('utf-8')

            if line.startswith("product/productId:"):
                product_id = line.split()[1]
                products.add(product_id)
                product_count[product_id] += 1
                if not product_id.startswith('B'):
                    logger += line + '\n'

            if line.startswith("review/userId:"):
                user_id = line.split()[1]
                if not user_id.startswith('A') and not user_id.startswith('un'):
                    logger += line + '\n'
                    continue
                user_record[user_id].add(product_id)

    # count # of distinct products bought by each user
    user_count = defaultdict(lambda: 0)
    for user, bought in user_record.items():
        user_count[user] = len(bought)
    
    # iterate through all user & products bought to generate output txt file
    with open("output.txt", 'w') as output_file:
        for user, bought in sorted(user_record.items(), key=lambda x: x[0]):
            output = user

            for product_id in sorted(list(bought)):
                output += ',' + product_id
            output += '\n'
            output_file.write(output)
    
    with open("logger.txt", 'w') as logger_file:
        logger_file.write(logger)

    # Q1 & Q2
    num_users = len(user_record.keys())
    num_products = len(products)

    # Q3 sort user_count by # of distinct products bought
    user_rank = sorted(user_count.items(), key=lambda x: (x[1], x[0]), reverse=True)

    # Q4 get median and find users meet the number
    median = sorted(user_count.values())[num_users//2]
    median_user = sorted([(key, value) for key, value in user_count.items() if value == median], key=lambda x: x[0])

    return num_users, num_products, user_rank[:3], median_user[:10] if len(median_user)>=10 else median_user

if __name__ == '__main__':
    time_start = time.time()
    filename = "Music.txt.gz"
    Q1, Q2, Q3, Q4 = parse(filename)
    time_end = time.time()
    print("Q1. How many unique users are there in the Music.txt.gz dataset?: {}".format(Q1))
    print("Q2. How many unique products are there in the Music.txt.gz dataset?: {}".format(Q2))
    print("Q3. What is the maximum number of products bought by a user?: {}".format(Q3))
    print("""Q4. What is the value of the median of the amount of products bought?
                 For those users having the median number of products, sort
                 their userIds in ascending lexicographic order. Then, print the first ten
                 userIds if there are more than ten or print all of them if not. \n{}""".format(Q4))
    print("Total running time is: {}".format(time_end-time_start))
