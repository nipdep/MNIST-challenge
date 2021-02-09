
num_list = [[1,2,3],[4,5,6],[7,8,9]]

def grid_parser(ls):
    width = len(ls[0])
    height = len(ls)
    fin_list = []
    sep_line = '+'+'+'.join(['-----',]*width)+'+'
    fin_list.append(sep_line)
    for line in range(height):
        line_lis = []
        for elem in range(width):
            if elem == 0:
                line_lis.append("{el:<5}".format(el=ls[line][elem]))
            elif elem == width-1:
                line_lis.append("{el:>5}".format(el=ls[line][elem]))
            else:
                line_lis.append("{el:^5}".format(el=ls[line][elem]))
        wh_line = '|'+'|'.join(line_lis)+'|'
        fin_list.append(wh_line)
        fin_list.append(sep_line)
    for i in fin_list:
        print(i)
    
grid_parser(num_list)