from re import split
# from functools import filter
from json import loads 
from pandas import DataFrame
from os import getcwd as pwd
from datetime import datetime
# Just use pandas!
# https://www.geeksforgeeks.org/how-to-randomly-select-rows-from-pandas-dataframe/

# For extracting things
def extract1(row):
    return row.map(extract2)
def extract2(val):
    if isinstance(val, list) and len(val) == 1:
        return val[0]
    return val

def createTable(data, categories=None, groupby = None, endocap=None):
    keys = sorted(data.columns.to_list())
    if categories is None:
        print("Pick Column/Row")
        # Assuming all entries have the same categories
        for index, key in enumerate(keys):
            print("{}: {}".format( index, key))
        categories = input("")
        # print(categories)
        categories = split("[\\s\:\\+\\,]", categories)
        categories = list(map(lambda x: keys[int(x)], categories))
    # Remove extra keys
    otherKeys = keys[:]
    del otherKeys[otherKeys.index(categories[0])]
    del otherKeys[otherKeys.index(categories[1])]

    if groupby is None:
                # key != groupby, keys)
        print("Pick a term to group by")
        for index, key in enumerate(otherKeys):
            print("{}: {}".format( index, key))
        groupby = otherKeys[int(input("").strip())]

    print("Build table with: {}".format(categories))
    print("Grouped by {}".format(groupby))
    del otherKeys[otherKeys.index(groupby)]
    del otherKeys[otherKeys.index('time')]
    del otherKeys[otherKeys.index('acc')]
    del otherKeys[otherKeys.index('tacc')]
    defaults = {}
    if len(otherKeys) != 0:
        for key in otherKeys:
            # Get the unique values for the unused key
            # So that we can generate a table based on just one of the values
            u = data[key].unique()
            # print(u)
            u.sort()
            if len(u) > 1:
                print("Please choose default {} value by index for creation".format(key))
                for index, val in enumerate(u):
                    print("{}: {}".format(index, u[index]))
                s = int(input("").strip())
                print("Selected:{} -- {}".format(s, u[s]))
                defaults[key] = u[int(s)]
            elif len(u) == 1:
                defaults[key] = u[0]
            else:
                print("{} has no values!".format(u))
                # print("defaults", defaults)

    alltables1 = []
    alltables2 = []
    table1 = "% {} -- {}\n%{}\n".format(datetime.now(), pwd(), defaults)
    table2 = "% {} -- {}\n%{}\n".format(datetime.now(), pwd(), defaults)
    # Redo everything using only the above variables!
    # Sample table -- categories[0] == vX, categories[1] == dX, groupby
    # determines subtable values
    # [     ][v1][v2][v2]
    # [   d1][f1][f2][f3]
    # [   d2][f4][f5][f6]
    # [   d3][f7][f8][f9]
    sdata = data.copy()
    for key in defaults.keys():
        # Get rid of all the data which does not have the required defaults
        sdata = sdata.loc[sdata[key] == defaults[key]]
    # Generate the first pandas table using pivots for time
    # print(sdata)
    tabs_t = []
    tabs_a = []
    groupby_vals = sorted(sdata[groupby].unique())
    for val in groupby_vals:
        # print(sdata.loc[sdata[groupby] == val])
        tabs_t.append(sdata.loc[sdata[groupby] == val].pivot(index=categories[1], 
            columns=categories[0], values='time'))
        tabs_a.append(sdata.loc[sdata[groupby] == val].pivot(index=categories[1], 
            columns=categories[0], values='acc'))
    # Generate all header and caption data for the table
    categories = list(map(lambda x: x.replace("_", " "), categories))
    groupby = groupby.replace("_", " ")
    defs = list(map(lambda x: "{} {}".format(x, defaults[x]).replace("_", " "),
        sorted(defaults.keys())))
    # The hardest part is building the table header :(
    width = len(tabs_t[0].columns.to_list()) 
    table1 += "\\caption{" +"Timing for {} versus {} grouped by {} with {} {}.".format(
            categories[0], categories[1], groupby, ", ".join(defs), endocap) \
    + "}\n\\vspace{0.5\\baselineskip" + "}\n" + "\\begin{tabular" \
    + "}{r"+ "{}".format("c"*width) + "}\\hline\n"
    table2 += "\\caption{" +"Timing for {} versus {} grouped by {} with {} {}.".format(
            categories[0], categories[1], groupby, ", ".join(defs), endocap) \
    + "}\n\\vspace{0.5\\baselineskip" + "}\n" + "\\begin{tabular" \
    + "}{r"+ "{}".format("c"*width) + "}\\hline\n"
    def help1(v, t="time"):
        v = v.split(",")
        v[0] = "{:16}".format(v[0].strip())
        for i, val in enumerate(v[1:]):
            if t == "time":
                if val:
                    v[i+1] = " {:8}".format(v[i+1].split(".")[0])
                else: # Empty string
                    v[i+1] = "       N/A"
            else:
                if val:
                    v[i+1] = "{:9.2f}".format(float(v[i+1])*100)
                else: # Empty string
                    v[i+1] = "    N/A"
        return v

    for i in range(len(tabs_t)):
        tab = tabs_t[i].to_csv().strip().split("\n")
        # print(tab)
        table1 += "{} {}".format(groupby_vals[i], groupby.title()) \
        + " & \\multicolumn{" + "{}".format(width)  +"}{c}{" \
        + "{}".format(categories[0]).title() + "}\\\\\n" 
        subhead = tab[0].split(",")
        subhead = "{:16} &".format(subhead[0]).title().replace("_", " ") \
                + "&".join(list(map(lambda x: " {:8} ".format(x), subhead[1:])))
        table1 += subhead + "\\\\\n\\hline\n"
        table1 += " \\\\\n".join(list(map(
            lambda row: " &".join(help1(row, "time")),
            tab[1:])))
        if i < len(tabs_t) - 1:
            table1 += "\\\\ \n\\hline\\hline\n"
        else:
            table1 += " \\\\ \n\\hline"
        # Accuracy Table
        tab = tabs_a[i].to_csv().strip().split("\n")
        # print(tab)
        table2 += "{} {}".format(groupby_vals[i], groupby.title()) \
        + " & \\multicolumn{" + "{}".format(width)  +"}{c}{" \
        + "{}".format(categories[0]).title() + "}\\\\\n" 
        subhead = tab[0].split(",")
        subhead = "{:16} &".format(subhead[0]).title().replace("_", " ") \
                + "&".join(list(map(lambda x: " {:8} ".format(x), subhead[1:])))
        table2 += subhead + "\\\\\n\\hline\n"
        table2 += " \\\\\n".join(list(map(
            lambda row: " &".join(help1(row, "acc")),
            tab[1:])))
        if i < len(tabs_a) - 1:
            table2 += " \\\\ \n\\hline\\hline\n"
        else:
            table2 += " \\\\\n\\hline"
        # print(tabs_a[i].to_csv().strip().split("\n"))
    # print(table2)
    table1 += "\n\end{tabular}"
    table2 += "\n\end{tabular}"
    return table1, table2

def loadData(filename):
    with open(filename, 'r') as infile:
        data = infile.read().strip()
    if not data.strip():
        print("File is empty: {}".format(filename))
        exit()
    data = list(map(lambda x: loads(x.replace("'", '"')), data.split("\n")))
    data = DataFrame(data)
    data = data.apply(extract1)
    return data


def saveOutput(ofilename, table):
    with open(ofilename, 'w') as outfile:
        outfile.write(table.replace("gpu", "GPUs").replace("Gpu", "GPUs"))

if __name__ == "__main__":
    from sys import argv
    if "-c" in argv:
        index = argv.index('-c') + 1
        categories = argv[index : index + 2]
    else:
        categories = None
    if "-g" in argv:
        groupby = argv[argv.index('-g') + 1]
    else:
        groupby = None
    if "-eoc" in argv:
        eoc = argv[argv.index('-eoc') + 1]
    else:
        eoc = None

    # Take in the json file to avoid a ton of work in the table builder
    # try:
    if "-f" in argv:
        filename = argv[argv.index('-f') + 1]
        data = loadData(filename)
        table1, table2 = createTable(data, categories=categories,
                groupby=groupby, endocap=eoc)
    else:
        print("Not enough information provided!")
        print("python script.py -f result.txt -j associatedJSON.json")
        exit()
    if '-o' in argv:
        ofilename1 = argv[argv.index('-o') + 1]
        ofilename2 = argv[argv.index('-o') + 2]
        saveOutput(ofilename1, table1)
        saveOutput(ofilename2, table2)
    else:
        saveOutput("./timings.tex", table1)
        saveOutput("./accuracy.tex", table2)
    # except Exception as e:
        # print(e)
