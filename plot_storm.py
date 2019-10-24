import matplotlib.pyplot as plt
from os.path import abspath as apath
from os.path import split as psplit
from os.path import basename
from numpy import load
from numpy import concatenate as cat
from numpy import split as aSplit
from multiprocessing import Pool
def plot_storm(title, oname, storm, vmin=-10,vmax=85):
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(storm[:, :, 0], cmap="gist_ncar", vmin=vmin, vmax=vmax)
    # plt.colorbar()
    plt.quiver(storm[:, :, 1], storm[:, :, 2])
    # plt.contour(all_out_data[rot_ex, :, :, 0])
    plt.title(title)
    plt.savefig(oname)
    plt.close()

def getMinMax(data):
    if not isinstance(data, list):
        channels = data[0].shape[-1]

    else:
        channels = data[0].shape[-1]
        # print(len(data))
        allMax = map(lambda storm:
        list(map(lambda channel: storm[...,channel].max(), range(channels))), 
            data)
        allMax = list(allMax)
        # print(len(allMax))
        allMax = list(zip(*allMax))
        # print(allMax[0])
        allMax = list(map(lambda channel: max(allMax[channel]),
            range(channels)))
        # For the minimum
        allMin = map(lambda storm:
            list(map(lambda channel: storm[...,channel].min(), range(channels))), 
                 data)
        allMin = list(zip(*list(allMin)))
        # # print(allMax[0])
        allMin = list(map(lambda channel: min(allMin[channel]),
            range(channels)))
        # print(len(allMax))
        # print(list(allMin))
        print(allMax, flush=True)
        print(allMin, flush=True)
    return allMin, allMax

def getOutputName(index, allNames):
    # allNames = [[161k, partialName], [10, partialName],...)
    partialIndex = index
    for count, name in allNames:
        partialIndex -= count
        if partialIndex <= 0:
            outname = "{}.{:07d}.png".format(name, partialIndex + count)
            break
    return outname

class poolObject:
    def __init__(self, allNames, cmin, cmax):
        # self.data = data
        self.allNames = allNames
        self.cmin = cmin
        self.cmax = cmax
    # def poolFunction(self, index):
    def poolFunction(self, val):
        # print("enter!!")
        index, data = val
        # plot_storm(title, oname, storm, vmin=-10,vmax=85):
        outname = getOutputName(index, allNames)
        # print("{}".format(outname))
        print(index, outname, data[0,...].shape, cmin, cmax)
        plot_storm("", outname, data[0,...],
                vmin=cmin,vmax=cmax)
        print("Saved {}".format(outname))

def poolFunction(allNames, cmin, cmax, val):
    # print("enter!!")
    index, data = val
    # plot_storm(title, oname, storm, vmin=-10,vmax=85):
    outname = getOutputName(index, allNames)
    # print("{}".format(outname))
    # print(index, outname, data[0,...].shape, cmin, cmax)
    plot_storm("", outname, data[0,...],
            vmin=cmin,vmax=cmax)
    print("Saved {}".format(outname), flush=True)


if __name__ == "__main__":
    import mpi4py
    from mpi4py import MPI
    from sys import argv
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if "-f" in argv:
        iname = argv[argv.index("-f") + 1]
        if "-o" in argv:
           oname =  argv[argv.index("-o") + 1]
        else:
            oname = "output.png"
        if "-i" in argv:
            index = int(argv[argv.index("-i") + 1])
        else:
            index = 0

        if "-t" in argv:
            title = argv[argv.index("-t") + 1]
        else:
            title = ""
        data = load(iname)
        cmin, cmax = getMinMax(data)

        plot_storm(title, oname, data[index,...], vmin=cmin[0], vmax=cmax[0])
    elif "-b" in argv:
        if "-o" in argv:
            # This means output to a folder
            outputTo = argv[argv.index("-o") + 1]  + "/"
        else:
            outputTo = "./"
        if "-p" in argv:
            totalP = int(argv[argv.index("-p") + 1])
        else:
            totalP = 8
        allFiles = argv[argv.index("-b") + 1:]
        allFiles = list(map(lambda fname: apath(fname), allFiles))
        allData  = []
        allNames = []
        print(len(allFiles), flush=True)
        for dataFile in allFiles:
            data = load(dataFile)
            newFilename = basename(dataFile)
            # p, newFilename = psplit(dataFile)
            # print(dataFile, p, newFilename)
            # print(newFilename)
            newFilename = newFilename.split(".")
            newFilename = ".".join(newFilename[:-1])
            outputName = "{}{}".format(outputTo, newFilename)
            allNames.append([data.shape[0], outputName])
            allData.extend(aSplit(data, data.shape[0]))
        # print(len(allData), allData[0].shape)
        # Combine all data
        # Instead just split them into individual things
        # Then do the minmax over those things
        # data = cat(allData)
        cmin, cmax = getMinMax(allData)
        # poolFunction(data, allNames, cmin, cmax, index):
        indices = list(range(len(allData)))
        vals = list(zip(indices, allData))
        # print("len of vals =", len(vals))
        print("Ready to start!", flush=True)
        for i in range(rank, len(allData), size):
            poolFunction(allNames, cmin[0], cmax[0], vals[i])
        # pObj = poolObject(allNames, cmin[0], cmax[0])
        # f = lambda index: poolFunction(data, allNames, cmin[0], cmax[0],
                # index)
        # p = Pool(totalP)
        # p.map(pObj.poolFunction, vals)
        # for index in range(data.shape[0]):
            # plot_storm("", outname, data[index,...],
                    # vmin=cmin[0], vmax=cmax[0])
            # print("Saved {}".format(outname))
    else:
        print("No file selected. Nothing to do...")
