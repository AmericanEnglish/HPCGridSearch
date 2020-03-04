from re import split
# from functools import filter
from json import load as loadf
from json import loads 

from os import getcwd as pwd
from datetime import datetime

def makeUnique(someList):
    return list(set(someList))

def createTable(data, jdata, categories=None, groupby = None, endocap=None):
    keys = sorted(jdata.keys())
    if categories is None:
        print("Pick Column/Row")
        # Assuming all entries have the same categories
        keys = sorted(jdata.keys())
        for index, key in enumerate(keys):
            print("{}: {}".format( index, key))
        categories = input("")
        # print(categories)
        categories = split("[\\s\:\\+\\,]", categories)
        categories = list(map(lambda x: keys[int(x)], categories))

    otherKeys = list(filter(lambda key: key != categories[0] and
            key != categories[1], keys))# and

    if groupby is None:
                # key != groupby, keys)
        print("Pick a term to group by")
        for index, key in enumerate(otherKeys):
            print("{}: {}".format( index, key))
        groupby = otherKeys[int(input("").strip())]

    print("Build table with: {}".format(categories))
    print("Grouped by {}".format(groupby))
    otherKeys = list(filter(lambda key: key != groupby, otherKeys))
    defaults = {}
    if len(otherKeys) != 0:
        # Be sure that there are multiple values
        for key in otherKeys:
            if len(jdata[key]) > 1:
                print("Please choose default {} value by index for creation".format(key))
                for index, val in enumerate(jdata[key]):
                    print("{}: {}".format(index, jdata[key][index]))
                defaults[key] = jdata[key][int(input("").strip())]
            else:
                defaults[key] = jdata[key][0]
                # print("defaults", defaults)
    # Filter out all dicts which don't meet the defaults
    # print(data)
    usableDicts = data
    for key in defaults.keys():
        usableDicts = list(filter(lambda aDict: defaults[key] == aDict[key],
                usableDicts))
    # Table 1 is time and table 2 is accuracy
    alltables1 = []
    alltables2 = []
    table1 = "% {} -- {}\n%{}\n".format(datetime.now(), pwd(), defaults)
    table2 = "% {} -- {}\n%{}\n".format(datetime.now(), pwd(), defaults)
    # Autogenerate the caption too
    table1 += "\\caption{Wall time for " 
    table1 += "{} versus {} grouped by {} with ".format(categories[0],
            categories[1], groupby).lower().replace("_", " ")
    table2 += "\\caption{Accuracy for " 
    table2 += "{} versus {} grouped by {} with ".format(categories[0],
            categories[1], groupby).lower().replace("_", " ")

    # if len(defaults.keys()) == 1:
        # table1 += " with a fixed value for "
        # table2 += " with a fixed value for "
    # else:
        # table1 += " with fixed values for "
        # table2 += " with fixed values for "
    # table1 += "".format(", ".join(sorted(defaults.keys())).lower())
    # The fixed default portion might need to come out eventually
    # This is also a special way to compute the generic string for
    # a fixed value for {variable}, {variable}  of {value}, {value} respectively.
    # table1 += "{}".format(", ".join(
        # sorted(defaults.keys()))).lower().replace("_", " ")
    # table2 += "{}".format(", ".join(
        # sorted(defaults.keys()))).lower().replace("_", " ")
    table1 +=  "{}".format(", ".join(
        list(map(lambda x: "{} {}".format(x, defaults[x]),
            sorted(defaults.keys()))))).replace("_", " ")
    table2 +=  "{}".format(", ".join(
        list(map(lambda x: "{} {}".format(x, defaults[x]),
            sorted(defaults.keys()))))).replace("_", " ")

    # table1 += "{}".format(", ".join(
        # list(map(lambda x: str(defaults[x]), sorted(defaults.keys()))))) 
    # table2 += "{}".format(", ".join(
        # list(map(lambda x: str(defaults[x]), sorted(defaults.keys()))))) 
    if eoc is None:
        table1 += "."
        table2 += "."
    else:
        table1 += " {}.".format(eoc)
        table2 += " {}.".format(eoc)
    # The end of the caption
    table1 +="}\n\\vspace{0.5\\baselineskip}\n"
    table2 +="}\n\\vspace{0.5\\baselineskip}\n"
    # Include the tabular 
    table1 += "\\begin{tabular}{r" 
    table1 +=  "c" * (len(jdata[categories[1]]))
    table1 +=  "} \\hline\n"
    table2 += "\\begin{tabular}{" \
            + "r{}".format("c"*(len(jdata[categories[1]]))) \
            + "} \\hline\n"
    # print("usableDicts", usableDicts)
    for group in jdata[groupby]:
        body1 = ""
        body2 = ""
        header = ""
        # header += "\\multicolumn{"+ "{}".format(len(jdata[categories[1]]) + 1) \
                # + "}{l}{" + "{} {}".format(group, groupby) + "}" + "\\\\\n"
        header += "{} {}".format(group, groupby).title().replace("_"," ") \
                + " & \\multicolumn{" \
                + "{}".format(len(jdata[categories[1]]) )  \
                + "}{c}{" + "{}".format(categories[1]).title().replace("_"," ") + "}\\\\\n"
                # + "}{l}{" + "{}/{}".format(*categories).title().replace("_"," ") + "}\\\\\n"
        # header += "\\hline\n"
        header += "{:15} ".format(categories[0]).title().replace("_", " ")
        header += "".join(list(map(lambda val: "& {:8}".format(val), 
            map(lambda val: str(val), sorted(jdata[categories[1]])))))
        header += "\\\\\n\\hline\n"
        body1 += header
        body2 += header
        groupDicts = list(filter(lambda aDict: aDict[groupby] == group,
            usableDicts))
        # print("groupDicts", groupDicts)
        for catval in jdata[categories[0]]:
            row1, row2 = createRow(jdata, categories, catval, groupDicts)
            body1 += row1
            body2 += row2
        # table1 += "\\hline\\hline\n"
        alltables1.append(body1)
        # table2 += "\\hline\\hline\n"
        alltables2.append(body2)
    table1 += "\\hline\\hline\n".join(alltables1) + "\\hline\n\\end{tabular}"
    table2 += "\\hline\\hline\n".join(alltables2) + "\\hline\n\\end{tabular}"
    return table1, table2

def createRow(jdata, categories, catval, dicts):
    # print(categories)
    # print(catval)
    # print("dicts", dicts)
    times = {}
    acc = {}
    columns = list(filter(lambda entry: entry[categories[0]] == catval, dicts))
    # print("columns",columns)
    # columns = list(
    for entry in columns:
        times[entry[categories[1]]] = entry['time']
        acc[entry[categories[1]]] = entry['acc']
    # map(lambda entry:
            # putInto(times, entry[categories[1]], entry['time']), columns)
    # map(lambda entry:
            # putInto(acc,   entry[categories[1]], entry['acc']),  columns)
        # s[entry[categories[0]]] = entry['time'], columns)# )
    row1 = "{:15} ".format(catval)
    row2 = "{:15} ".format(catval)
    # print(acc)
    # print(times, acc)
    for val in jdata[categories[1]]:
        if val in times.keys() and bool(times[val]):
            row1 += "& {:8}".format(times[val])
            row2 += "& {:8}".format(acc[val])
        else:
            row1 += "& {:8}".format("N/A")
            row2 += "& {:8}".format("N/A")

    row1 += "\\\\\n"
    row2 += "\\\\\n"
    return row1, row2

def putInto(aDict, key, val):
    aDict[key] = val
    return True

def parseLine(line):
    if "::" in line:
        # entries = line.split("::")[::2]
        entries = line.split("::")
        entries = list(map(lambda entry: entry.strip(), entries))
        # entries = [entries[0].strip(), entries[1].strip()]
        # Cut off the "." in the time entry
        # print(entries)
        ptime = entries[2].split(".")[0]
        acc = entries[1].split("->")[1].strip()
        # Create a dictionary from the first entry
        aDict = loads(entries[0].replace("'","\""))
        aDict['time'] = ptime
        aDict['acc'] = "{:6.2f}".format(float(acc)*100)
        # print(aDict)
    else:
        # aDict = loads(entries[0].replace("'","\""))
        line = line.replace("'",'"')
        aDict = loads(line)
        # Cut off fractional seconds
        aDict['time'] = aDict['time'].split(".")[0]
        # Converts accuracy to a percent
        try:
            aDict['acc'] = "{:6.2f}".format(float(aDict['acc'])*100)
        except ValueError:
            aDict['acc'] = str(aDict['acc']*100).split(".")[0]
    return aDict

    
def loadData(filename, jfilename):
    with open(jfilename, 'r') as infile:
        jdata = infile.read().strip()
    with open(filename, 'r') as infile:
        data = infile.read().strip()
    if not data.strip():
        print("File is empty: {}".format(filename))
        exit()
    elif not jdata.strip():
        print("jFile is empty: {}".format(jfilename))
        exit()
    jdata = loads(jdata)
    data = list(map(parseLine, data.split("\n")))
    for aDict in data:
        keys = aDict.keys()
        for key in keys:
            if key != 'time' and key != 'acc':
                aDict[key] = aDict[key][0]
    # print(data)
    return data, jdata


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
    try:
        if "-f" in argv and "-j" in argv:
            filename = argv[argv.index('-f') + 1]
            jfilename = argv[argv.index('-j') + 1]
            data, jdata = loadData(filename, jfilename)
            table1, table2 = createTable(data, jdata, categories=categories,
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
    except Exception as e:
        print(e)
