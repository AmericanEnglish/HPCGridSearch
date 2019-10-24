from functools import reduce
def generateAllCombinations(args):
    """(dict) -> list of dicts
    
    Takes in a dictionary that contains ranges of training parameters.
    Produces all possible combinations of those training parameters."""
    return recursiveComboGeneration(0, args, [])

def recursiveComboGeneration(i, params, allDicts):
    """ (int, list of strings, list of dicts) -> (list of dicts)

    Takes in an integer which represents the index of the sorted(params.keys()) index.
    Recursively generated all possible dictionarys of all combinations of the provided parameters.
    """
    allKeys = sorted(params.keys())
    if i == 0:
        name = allKeys[i]
        # Sets up the initial configuration
        allDicts =  map(lambda val: putInto({}, name, val), params[name])
        allDicts = flattenList(allDicts)
        # print(name, allDicts)
        return recursiveComboGeneration(i+1, params, allDicts)
    elif i < len(allKeys):
        name = allKeys[i]
        partial = lambda sDict: list(map(lambda val: putInto(sDict, name, val), params[name]))
        # Generates a list of dicts for every dict in the list
        if isinstance(params[name], list):
            # allDicts = map(lambda aDict: map(lambda val: putInto(aDict, name, val), params[name]), allDicts)
            # print(allDicts)
            allDicts = map(lambda aDict: partial(aDict), allDicts)

        else:
            # allDicts = map(lambda aDict: map(lambda val: putInto(aDict, name, val), list(params[name])), allDicts)
            allDicts = map(lambda aDict: minimap(aDict), allDicts)
        allDicts = flattenList(allDicts)
        # print(name, allDicts)
        return recursiveComboGeneration(i+1, params, allDicts)
    else:
        allDicts = flattenList(allDicts)
        # print(allDicts)
        return flattenList(allDicts)

def putInto(aDict, key, val):
    aDict = aDict.copy()
    # print(aDict, key, val)
    # Keras requires an iterable item
    aDict[key] = [val]
    return aDict

def isFlat(someList):
    if len(someList) > 1:
        return reduce(lambda x,y: 
            (not isinstance(x, list) and not isinstance(x, map)) and
            (not isinstance(y, list) and not isinstance(y, map)), someList)
    # This coveres the edge case of [[1]] which is empty but NOT flat
    elif len(someList) == 1:
        return not isinstance(someList[0], list)
    else:
        return True

def flattenList(someList):
    if isinstance(someList, map):
        someList = list(someList)
    while not isFlat(someList):
        newList = []
        for item in someList:
            if isinstance(item, map):
                newList.extend(list(item))
            elif isinstance(item, list):
                newList.extend(item)
            else:
                newList.append(item)
        someList = newList
    return someList

def getMaxCombos(params):
    return reduce(lambda x, y: x*y, (map(lambda key: len(params[key]),
        params.keys())))

def comboGenerator(n, params={}):
    """(int, dictionary) -> dictionary

    Acts as a pseudo-generator. Given an index it returns a dictionary.
    If you were to generate a list of all possible combinations of keys
    in the dictionary then this would return the ith item in that list.

    While there is a lot more sorting, it should be very very fast especially
    when compared the time and memory it takes to fully generate the whole list
    of combos instead.
    """
    keys = sorted(params.keys(), reverse=True)
    maxCombos = getMaxCombos(params)
    # print(maxCombos)
    if n >= maxCombos:
        return None

    if "data_multiplier" in keys:
        del keys[keys.index("data_multiplier")]
    else:
        params["data_multiplier"] = [1]
    # keys.insert(0, "data_multiplier")
    keys.append("data_multiplier")
    result = {}
    for key in keys:
        val  = len(params[key])
        result[key] = [params[key][n % val]]
        n = n // val
    # return keys, totals
    return result
