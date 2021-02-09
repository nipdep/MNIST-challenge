from colored import fg
import random
import time

# Colors
green = fg("green")
white = fg("white")
yellow = fg("yellow")
blue = fg("blue")
magenta = fg("magenta")
red = fg('red')
wheat_4 = fg("wheat_4")
dark_slate_gray_1 = fg("dark_slate_gray_1")
purple_1b = fg("purple_1b")

# Main Assets
nums_2 = ['1', '2']
nums_3 = ['1', '2', '3']
nums_4 = ['1', '2', '3', '4']
nums_6 = ['1', '2', '3', '4', '5', '6']
revenue = 1000000
employee_num = 0
# Products
products = 0
phone_products = 0
laptop_products = 0
watch_products = 0
products_sold = 0
casual_clothing_products = 0
suits_products = 0
fashion_accessory_products = 0
shares_num = 0
shares_sold = 0

# Start with company's name
company_name = input(red + "What would you like to name your company: ")
time.sleep(1)

while True:
    company_specialization = input(dark_slate_gray_1  + "\nWhat specialization would you like for your company:\n\n         [1] Technology\n         [2] Clothing\n\n---> ")

    if company_specialization == '1':
        specialization = "Technology" 
        break   
    elif company_specialization == '2':
        specialization = 'Clothing'   
        break


def company():
    global employee_num
    global shares
    global products
    global phone_products, laptop_products, watch_products
    def employees():
        global employee_num 
        global revenue

        # getting file data
        employees_open = open('employees.txt', 'r')
        employees_open_read = employees_open.readlines()
        employees_open_read_fixed = [line.strip() for line in employees_open_read]
        employees_open = employees_open.close()

        #Employee names
        random_employee1 = random.choice(employees_open_read_fixed)
        random_employee2 = random.choice(employees_open_read_fixed) 
        random_employee3 = random.choice(employees_open_read_fixed) 

        # Salaries
        salaries = [10000, 15000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]
        employee_salary1 = random.choice(salaries)
        employee_salary2 = random.choice(salaries)
        employee_salary3 = random.choice(salaries)

        #Names in a list
        global revenue

        while True:
            hiring = input(f"{yellow}\nHere is a list of people you can hire:\n\n         [1] Name: {random_employee1} | Salary: ${employee_salary1}\n         [2] Name: {random_employee2} | Salary: ${employee_salary2}\n         [3] Name: {random_employee3} | Salary: ${employee_salary3}\n         [4] Or go back to main page. \n\nEnter the number of the person you would like to hire: ---> ")
            while hiring in nums_3:
                if hiring == '1':
                    revenue -= employee_salary1
                    random_employee1 = random.choice(employees_open_read_fixed)
                    employee_num += 1

                    break

                if hiring == '2':
                    revenue -= employee_salary2
                    random_employee2 = random.choice(employees_open_read_fixed)
                    employee_num += 1
                    break

                if hiring == '3':
                    revenue -= employee_salary3
                    random_employee3 = random.choice(employees_open_read_fixed)
                    employee_num += 1

                    break

            if hiring == '4':
                break

    def product_sales():
        def tech():
            global phone_products, laptop_products, watch_products, shares_num, employee_num, revenue, products
            if specialization == 'Technology':
                product_make = input("What product would you like to design in technology:\n\n         [1] Phone\n         [2] Laptops\n         [3] Watches\n         [4] Or go back.\n ---> ")
                    
    
                while True:
                    if product_make == '4':
                        break
                    if product_make == '1':
                        max_products = employee_num*4
                        if max_products == products:
                            print(red + "\nYou don't have enough employees. You can only design 4 products with one employee.\n")
                            break
                        while max_products != products:                           
                            try:  
                                print('\nPhone prices must be $500 to $7000 dollars.') 
                                print(purple_1b + "\nManufacturing Price: $400\nLabor Cost: $100") 
                                phone_selling_price = int(input('Enter the price: '))
                                if phone_selling_price > 500 and phone_selling_price < 7000:
                                    phone_products += 1
                                    print(phone_products)
                                    products += 1
                                    revenue -= 10000
                                    shares_num += 50
                                    break
                                break
                            except ValueError: 
                                # let the loop go again  
                                print(red + 'Invalid')
                                break
                        break 
                        
                    if product_make == '2':
                        max_products = employee_num*4
                        if max_products == products:
                            print(red + "\nYou don't have enough employees. You can only design 4 products with one employee.\n")
                            break
                        while max_products != products:                           
                            try:  
                                print('\nLaptop prices must be $1000 to $10,000 dollars.') 
                                print(purple_1b + "\nManufacturing Price: $700\nLabor Cost: $100") 
                                laptop_selling_price = int(input('Enter the price: '))
                                if laptop_selling_price > 1000 and laptop_selling_price < 10000:
                                    laptop_products += 1
                                    print(laptop_products)
                                    products += 1
                                    revenue -= 20000
                                    shares_num += 50
                                    break
                                break
                            except ValueError: 
                                # let the loop go again  
                                print(red + 'Invalid')  
                                break
                        break 
                    
                    if product_make == '3':
                        max_products = employee_num*4
                        if max_products == products:
                            print(red + "\nYou don't have enough employees. You can only design 4 products with one employee.\n")
                            break
                        
                        while max_products != products:                           
                            try:  
                                print('\nWatch prices must be $100 to $2,000 dollars.') 
                                print(purple_1b + "\nManufacturing Price: $200\nLabor Cost: $100") 
                                watch_selling_price = int(input('Enter the price: '))
                                if watch_selling_price > 100 and watch_selling_price < 2000:
                                    watch_products += 1
                                    print(watch_products)
                                    products += 1
                                    revenue -= 10000
                                    shares_num += 50
                                    break
                                break
                            except ValueError: 
                                # let the loop go again  
                                print(red + 'Invalid')
                                break
                        break
                
        tech()

        def clothing():
            global products
            global revenue
            global employee_num
            global casual_clothing_products, suits_products, fashion_accessory_products, shares_num
            if specialization == 'Clothing':
                product_make = input("What product would you like to design in clothing:\n\n         [1]  Casual Clothes\n         [2] Suits\n         [3] Fshion Accesory\n         [4] Or go back.\n ---> ")
                    
    
                while True:
                    if product_make == '4':
                        break
                    if product_make == '1':
                        revenue -= 10000
                        shares_num += 50
                        max_products = employee_num*4
                        if max_products == products:
                            print(red + "\nYou don't have enough employees. You can only design 4 products with one employee.\n")
                            break
                        while max_products != products:                           
                            try:  
                                print('\nWatch prices must be $50 to $100 dollars.') 
                                print(purple_1b + "\nManufacturing Price: $40\nLabor Cost: $10") 
                                casual_selling_price = int(input('Enter the price: '))
                                if casual_selling_price > 50 and casual_selling_price < 100:
                                    casual_clothing_products += 1
                                    print(casual_clothing_products)
                                    products += 1
                                    revenue -= 1000
                                    shares_num += 20
                                    break
                                break
                            except ValueError: 
                                # let the loop go again  
                                print(red + 'Invalid')
                                break
                        break 
                        
                    if product_make == '2':
                        revenue -= 10000
                        shares_num += 50
                        max_products = employee_num*4
                        if max_products == products:
                            print(red + "\nYou don't have enough employees. You can only design 4 products with one employee.\n")
                            break
                        while max_products != products:                           
                            try:  
                                print('\nSuit prices must be $70 to $120 dollars.') 
                                print(purple_1b + "\nManufacturing Price: $60\nLabor Cost: $15") 
                                suit_selling_price = int(input('Enter the price: '))
                                if suit_selling_price > 70 and suit_selling_price < 120:
                                    suits_products += 1
                                    print(suits_products)
                                    products += 1
                                    revenue -= 1000
                                    shares_num += 20
                                    break
                                break
                            except ValueError: 
                                # let the loop go again  
                                print(red + 'Invalid')
                                break
                        break 
                    
                    if product_make == '3':
                        revenue -= 10000
                        shares_num += 50
                        max_products = employee_num*4
                        if max_products == products:
                            print(red + "\nYou don't have enough employees. You can only design 4 products with one employee.\n")
                            break
                        while max_products != products:                           
                            try:  
                                print("\nFashion Accesories' prices must be $30 to $60 dollars.") 
                                print(purple_1b + "\nManufacturing Price: $20\nLabor Cost: $5") 
                                fashion_selling_price = int(input('Enter the price: '))
                                if fashion_selling_price > 30 and fashion_selling_price < 60:
                                    fashion_accessory_products += 1
                                    print(fashion_accessory_products)
                                    products += 1
                                    revenue -= 1000
                                    shares_num += 20
                                    break
                                break
                            except ValueError: 
                                # let the loop go again  
                                print(red + 'Invalid')
                                break
                        break

        clothing()

    def shares():
        global revenue
        global shares_num
        global shares_sold

        if shares_num <= 0:
            print(red + "\nBRUH, you don't have any sares. Make products to get some.\n")

        while shares_num > 0: 
          try:
            shares_how_many = int(input(f"{green}\nYou have {shares_num} shares.  Each shares prices at $100.\nDon't press enter or enter a letter because you will get an error.\nHow many shares do you want to sell: "))
            if shares_how_many <= shares_num:
              shares_num -= shares_how_many
              revenue += shares_how_many * 100
              shares_sold += shares_how_many
              break
          except ValueError:
            print("Invalid")
            break

    def sell_products():
        global revenue, phone_selling_price, phone_products
        if products == 0:
            print("You don't have any products.") 
        if specialization == 'Technology':
            print(green + f"\nYou have:\n{phone_products} Phone Products\n{laptop_products} Laptop Products\n{watch_products} Watch Products.\n")
            sell = input("\nWhat product do you want to sell:\n         [1] Phone\n         [2] Laptop\n         [3] Watches.\n--->")

            while sell in nums_3:
                
                if sell == '1':
                    if phone_products == 0:
                        print("You don't have any phone product")
                    if phone_products > 0:
                        while True:                           
                            try: 
                                 
                                how_many = int(input("How many products do you want to produce. Enter a number: "))
                                revenue += how_many * phone_selling_price
                                phone_products -= 1 
                                revenue -= 500*how_many 
                                break
                            except ValueError: 
                                # let the loop go again  
                                print(red + 'Invalid')
                                break
                        break 

    while True:
        first_choice = input(magenta + "\nWhat do you want to do:\n\n         [1] See your company's stats\n         [2] Work on your company.\n---> ")

        while first_choice in nums_3:
            if first_choice == '1':
                print(f"{green}Company Name: {company_name.title()}.INC\nSpecialization: {specialization}\nRevenue: {revenue}\nEmployees: {employee_num}\nProducts Made: {products}\nShares: {shares_num}\nShares Sold: {shares_sold}")
                break

            if first_choice == '2':
                company_work = input(wheat_4 + "Here is what you can do with your company:\n\n         [1] Hire new employees\n         [2] Build a product.\n         [3] Sell shares\n         [4] Sell your products\n         [5] Or go back.\n---> ")

                if company_work == '5':
                    break

                while company_work in nums_6:
                    if company_work == '1':
                        employees()
                        break

                    if company_work == '2' and employee_num > 0 :
                        product_sales()
                        break

                    elif company_work == '2' and employee_num == 0:
                        print(red + "\nYou don't have enough employees. Hire some.\n")
                        break

                    if company_work == '3' and revenue > 0:
                        shares()
                        break

                    elif company_work == '3' and revenue <= 0:
                        print(red + "\nYou don't have enough money. Sell your products to earn money.\n")
                        break
                    if company_work == '4':
                        sell_products()
                        break
                        
                        
                        


                


company()