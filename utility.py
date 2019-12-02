import numpy as np
import numpy.matlib as mat
from scipy import signal
import scipy as sci
import matplotlib.pyplot as plt
import random
import os
import shutil

class edgelink(object):
    """
     EDGELINK - Link edge points in an image into lists

    ********************************************************************************************************************
     Usage: [edgelist edgeim, etype] = edgelink(im, minlength, location)

        **Warning** 'minlength' is ignored at the moment because 'cleanedgelist' has some bugs and can be memory hungry

     Arguments:  im         - Binary edge image, it is assumed that edges have been thinned (or are nearly thin).

                 minlength  - Optional minimum edge length of interest, defaults to 1 if omitted or specified as [].
                              Ignored at the moment.

                 location   - Optional complex valued image holding subpixel locations of edge points. For any pixel the
                              real part holds the subpixel row coordinate of that edge point and the imaginary part holds
                              the column coordinate.  See NONMAXSUP.  If this argument is supplied the edgelists will be
                              formed from the subpixel coordinates, otherwise the the integer pixel coordinates of points
                              in 'im' are used.
    ********************************************************************************************************************
     Returns:  edgelist - a cell array of edge lists in row,column coords in the form
                         { [r1 c1   [r1 c1   etc }
                            r2 c2    ...
                            ...
                            rN cN]   ....]

               edgeim   - Image with pixels labeled with edge number. Note that junctions in the labeled edge image will be
                          labeled with the edge number of the last edge that was tracked through it.  Note that this image
                          also includes edges that do not meet the minimum length specification. If you want to see just the
                          edges that meet the specification you should pass the edgelist to DRAWEDGELIST.

                etype   - Array of values, one for each edge segment indicating its type
                          0  - Start free, end free
                          1  - Start free, end junction
                          2  - Start junction, end free (should not happen)
                          3  - Start junction, end junction
                          4  - Loop

    ********************************************************************************************************************
     This function links edge points together into lists of coordinate pairs.
     Where an edge junction is encountered the list is terminated and a separate
     list is generated for each of the branches.

        """
    def __init__(self, im, minilength):
        if 'im' in vars().keys():
            self.edgeim = im
        else:
            raise NameError('edgelink: Image is undefined variable.')  # Error. Stop.

        if minilength != 'Ignore':
            self.minilength = minilength
            # print('edgelink: Minimum length is % d.\n' % minilength)

    def get_edgelist(self):
        self.edgeim = (self.edgeim.copy() != 0)                    # Make sure image is binary
        self.clean = bwmorph(self.edgeim.copy(), 'clean')         # Remove isolated pixels
        self.thin = bwmorph(self.clean, 'thin')          # Make sure edges are thinned
        thin_float = self.thin.copy()
        self.row = self.edgeim.shape[0]
        self.col = self.edgeim.shape[1]
        [rj, cj], [re, ce] = findendsjunctions(thin_float)
        self.ej = [rj, cj], [re, ce]

        num_junct = len(rj)
        num_end = len(re)

        # Create a sparse matrix to mark junction locations. This makes junction testing much faster.  A value
        # of 1 indicates a junction, a value of 2 indicates we have visited the junction.
        data = np.asarray([1 for ind in range(num_junct)])

        if len(rj) != len(data) or len(rj) != len(cj):
            raise ValueError('edgelink: Junction size does not match.')
        junct = sci.sparse.coo_matrix((data, (rj, cj)), shape=(self.row, self.col))
#        print(junct)
        junct = junct.tolil()
        self.junct = junct.copy()

        thin_float = thin_float * 1.0
        edgeNo = -1

        """
        # Summary of strategy:
        # 1) From every end point track until we encounter an end point or junction.  As we track points along an
        # edge image pixels are labeled with the -ve of their edge No.
        # 2) From every junction track out on any edges that have not been labeled yet.
        # 3) Scan through the image looking for any unlabeled pixels. These correspond to isolated loops that have no
        # junctions.
        """

        edgelist = []
        etype = []
        # 1) Form tracks from each unlabeled endpoint until we encounter another endpoint or junction
        for idx in range(num_end):
            if thin_float[re[idx], ce[idx]] == 1:       # Endpoint is unlabeled
                edgeNo = edgeNo + 1
                tempedge, endType = trackedge(re[idx], ce[idx], edgeNo, 'Ignore', 'Ignore', 'Ignore', thin_float, self.junct)
                edgelist.append(tempedge)
                etype.append(endType)
        



        """
        2) Handle junctions.
        # Junctions are awkward when they are adjacent to other junctions.  We start by looking at all the neighbours
        # of a junction. If there is an adjacent junction we first create a 2-element edgetrack that links the two
        # junctions together.  We then look to see if there are any non-junction edge pixels that are adjacent to both
        # junctions. We then test to see which of the two junctions is closest to this common pixel and initiate an
        # edge track from the closest of the two junctions through this pixel. When we do this we set the
        # 'avoidJunction' flag in the call to trackedge so that the edge track does not immediately loop back and
        # terminate on the other adjacent junction. Having checked all the common neighbours of both junctions we then
        # track out on any remaining untracked neighbours of the junction

        """

        for j in range(num_junct):
            if self.junct[rj[j], cj[j]] != 2:       # We have not visited this junction
                self.junct[rj[j], cj[j]] = 2         # Now we have :)

                # Call availablepixels with edgeNo = 0 so that we get a list of available neighbouring pixels that can
                # be linked to and a list of all neighbouring pixels that are also junctions.
                [all_ra, all_ca, all_rj, all_cj] = availablepixels(rj[j], cj[j], 0, thin_float, self.junct)

                # For all adjacent junctions. Create a 2-element edgetrack to each adjacent junction.
                for k in range(len(all_rj)):
                    edgeNo = edgeNo + 1
                    edgelist.append(np.array([[rj[j], cj[j]], [all_rj[k], all_cj[k]]]))
                    etype.append(3)
                    thin_float[rj[j], cj[j]] = -edgeNo
                    thin_float[all_rj[k], all_cj[k]] = -edgeNo

                    # Check if the adjacent junction has some untracked pixels that are also adjacent to the initial
                    # junction. Thus we need to get available pixels adjacent to junction (rj(k) cj(k)).
                    [rak, cak, rbk, cbk] = availablepixels(all_rj[k], all_cj[k], 0, thin_float, self.junct)

                    # If both junctions have untracked neighbours that need checking...
                    if len(all_ra) != 0 and len(rak) != 0:
                        adj = np.asarray([all_ra, all_ca])
                        adj = adj.transpose()                   # adj[:,0] is row, adj[:,1] is col
                        akdj = np.asarray([rak, cak])
                        akdj = akdj.transpose()
                        adj_ind = {tuple(row[:]): 'None' for row in adj}
                        akdj_ind = {tuple(row[:]): 'None' for row in akdj}
                        commonrc = adj_ind.keys() and akdj_ind.keys()
                        commonrc = np.asarray(list(commonrc))       # commonrc[:, 0] is row, commonrc[:, 1] is col
                        for n in range(commonrc.shape[0]):
                            # If one of the junctions j or k is closer to this common neighbour use that as the start of
                            # the edge track and the common neighbour as the 2nd element. When we call trackedge we set
                            # the avoidJunction flag to prevent the track immediately connecting back to the other
                            # junction.
                            distj = np.linalg.norm([commonrc[n, 0], commonrc[n, 1]] - np.array([rj[j], cj[j]]))
                            distk = np.linalg.norm([commonrc[n, 0], commonrc[n, 1]] - np.array([all_rj[k], all_cj[k]]))
                            edgeNo = edgeNo + 1
                            if distj < distk:
                                tempedge, endType = trackedge(rj[j], cj[j], edgeNo, commonrc[n, 0], commonrc[n, 1], 1, thin_float, self.junct)
                                edgelist.append(tempedge)
                            else:
                                tempedge, endType = trackedge(all_rj[k], all_cj[k], edgeNo, commonrc[n, 0], commonrc[n, 1], 1, thin_float, self.junct)
                                edgelist.append(tempedge)
                            etype.append(3)

                    for m in range(len(rak)):
                        if thin_float[rak[m], cak[m]] == 1:
                            edgeNo = edgeNo + 1
                            tempedge, endType = trackedge(all_rj[k], all_cj[k], edgeNo, rak[m], cak[m], 'Ignore', thin_float, self.junct)
                            edgelist.append(tempedge)
                            etype.append(3)

                    self.junct[all_rj[k], all_cj[k]] = 2

                for m in range(len(all_ra)):
                    if thin_float[all_ra[m], all_ca[m]] == 1:
                        edgeNo = edgeNo + 1
                        tempedge, endType = trackedge(rj[j], cj[j], edgeNo, all_ra[m], all_ca[m], 'Ignore', thin_float, self.junct)
                        edgelist.append(tempedge)
                        etype.append(3)

        # 3) Scan through the image looking for any unlabeled pixels. These should correspond to isolated loops that
        # have no junctions or endpoints.
        for ru in range(self.row):
            for cu in range(self.col):
                if thin_float[ru, cu] == 1:
                    edgeNo = edgeNo + 1
                    tempedge, endType = trackedge(ru, cu, edgeNo, 'Ignore', 'Ignore', 'Ignore', thin_float, self.junct)
                    if endType != 0:
                        edgelist.append(tempedge)
                        etype.append(endType)

        neg_image = -thin_float.copy()
        thin_float = neg_image
#        if hasattr(self, 'minilength'):
#            edgelist = cleanedgelist(edgelist, self.minilength)

        # if hasattr(self, 'minilength'):
        #     edgelist = cleanedgelist(edgelist, self.minilength)

        self.edgelist = edgelist
        self.etype = etype
        return


def trackedge(rstart, cstart, edgeNo, r2, c2, avdjunct, edgeim, junct):
    """
    EDGELINK - Link edge points in an image into lists

    ********************************************************************************************************************
    Function to track all the edge points starting from an end point or junction. As it tracks it stores the coords of
    the edge points in an array and labels the pixels in the edge image with the -ve of their edge number. This
    continues until no more connected points are found, or a junction point is encountered.

    ********************************************************************************************************************
    Usage:   edgepoints = trackedge(rstart, cstart, edgeNo, r2, c2, avdjunct, edgeim)

    Arguments:   rstart, cstart   - Row and column No of starting point.

                 edgeNo           - The current edge number.

                 r2, c2           - Optional row and column coords of 2nd point.

                 avoidJunction    - Optional flag indicating that (r2,c2) should not be immediately connected to a
                 junction (if possible).

                 edgeim           - Thinned edge image
                 junct            - Junctions map 

    ********************************************************************************************************************
    Returns:     edgepoints       - Nx2 array of row and col values for each edge point.

                 endType          - 0 for a free end    1 for a junction    5 for a loop

    ********************************************************************************************************************
    """
    if avdjunct == 'Ignore':
        avdjunct = 0
    edgepoints = np.array([rstart, cstart])         # Start a new list for this edge.
    edgepoints = np.reshape(edgepoints, [1, 2])
    edgeim[rstart, cstart] = -edgeNo                # Edge points in the image are encoded by -ve of their edgeNo.
    preferredDirection = 0                          # Flag indicating we have/not a preferred direction.

    # If the second point has been supplied add it to the track and set the path direction
    if r2 != 'Ignore' and c2 != 'Ignore':
        addpoint = np.array([r2, c2])
        addpoint = np.reshape(addpoint, [1, 2])
        edgepoints = np.vstack((edgepoints, addpoint))    # row = edge[:, 0], col = edge[:, 1]
        edgeim[r2, c2] = -edgeNo

        # Initialise direction vector of path and set the current point on the path
        dirn = unitvector(np.array([r2 - rstart, c2 - cstart]))
        r = r2
        c = c2
        preferredDirection = 1
    else:
        dirn = np.array([0, 0])
        r = rstart
        c = cstart

    # Find all the pixels we could link to
    [ra, ca, rj, cj] = availablepixels(r, c, edgeNo, edgeim, junct)

    while len(ra) != 0 or len(rj) != 0:
        #First see if we can link to a junction. Choose the junction that results in a move that is as close as possible
        # to dirn. If we have no preferred direction, and there is a choice, link to the closest junction
        # We enter this block:
        # IF there are junction points and we are not trying to avoid a junction
        # OR there are junction points and no non-junction points, ie we have to enter it even if we are trying to
        # avoid a junction
        if (len(rj) != 0 and not avdjunct) or (len(rj) != 0 and len(ra) == 0):
            # If we have a preferred direction choose the junction that results in a move that is as close as possible
            # to dirn.
            if preferredDirection:
                dotp = -np.inf
                for idx in range(len(rj)):
                    dirna = unitvector(np.array([rj[idx] - r, cj[idx] - c]))
                    dp = np.sum(dirn * dirna)
                    if dp > dotp:
                        dotp = dp
                        rbest, cbest = rj[idx], cj[idx]
                        dirnbest = dirna
            else:
                # Otherwise if we have no established direction, we should pick a 4-connected junction if possible as
                # it will be closest.  This only affects tracks of length 1 (Why do I worry about this...?!).
                distbest = np.inf
                for idx in range(len(rj)):
                    dist = np.sum(np.abs(np.array([rj[idx] -r, cj[idx] -c])))
                    if dist < distbest:
                        rbest, cbest = rj[idx], cj[idx]
                        distbest = dist
                        dirnbest = unitvector(np.array([rj[idx] - r, cj[idx] - c]))
                preferredDirection = 1
        else:
            # If there were no junctions to link to choose the available non-junction pixel that results in a move
            # that is as close as possible to dirn
            dotp = -np.inf
            for idx in range(len(ra)):
                dirna = unitvector(np.array([ra[idx] - r, ca[idx] - c]))
                dp = np.sum(dirn * dirna)
                if dp > dotp:
                    dotp = dp
                    rbest, cbest = ra[idx], ca[idx]
                    dirnbest = dirna
            avdjunct = 0    # Clear the avoidJunction flag if it had been set

        # Append the best pixel to the edgelist and update the direction and EDGEIM
        r, c = rbest, cbest
        addpoint = np.array([r, c])
        addpoint = np.reshape(addpoint, [1, 2])
        edgepoints = np.vstack((edgepoints, addpoint))    # row = edge[:, 0], col = edge[:, 1]
        dirn = dirnbest
        edgeim[r, c] = -edgeNo

        # If this point is a junction exit here
        if junct[r, c]:
            endType = 1
            return edgepoints, endType
        else:
            [ra, ca, rj, cj] = availablepixels(r, c, edgeNo, edgeim, junct)

    # If we get here we are at an endpoint or our sequence of pixels form a loop.  If it is a loop the edgelist should
    # have start and end points matched to form a loop.  If the number of points in the list is four or more (the
    # minimum number that could form a loop), and the endpoints are within a pixel of each other, append a copy of the
    # first point to the end to complete the loop

    endType = 0     # Mark end as being free, unless it is reset below

    if len(edgepoints) >= 4:
        if abs(edgepoints[0, 0] - edgepoints[-1, 0]) <= 1 and abs(edgepoints[0, 1] - edgepoints[-1, 1]) <= 1:
            edgepoints = np.vstack((edgepoints, edgepoints[0, :]))
            endType = 5

    return edgepoints, endType


def cleanedgelist(edgelist, minlength):
    """
    CLEANEDGELIST - remove short edges from a set of edgelists
    
    ********************************************************************************************************************
    Function to clean up a set of edge lists generated by EDGELINK so that isolated edges and spurs that are shorter 
    that a minimum length are removed. This code can also be use with a set of line segments generated by LINESEG.
    
    Usage: nedgelist = cleanedgelist(edgelist, minlength)

    ********************************************************************************************************************    
    Arguments:
                edgelist - a cell array of edge lists in row,column coords in
                           the form
                           { [r1 c1   [r1 c1   etc }
                              r2 c2    ...
                                       ...
                              rN cN]   ....]   
                minlength - minimum edge length of interest

    ********************************************************************************************************************
    Returns:
                nedgelist - the new, cleaned up set of edgelists

    """
    Nedges = len(edgelist)
    Nnodes = 2 * Nedges
    # Each edgelist has two end nodes - the starting point and the ending point. We build up an adjacency/connection
    # matrix for each node so that we can determine which, if any, edgelists are connected to a node. We also maintain
    # an adjacency matrix for the edges themselves.
    # It is tricky maintaining all this information but it does allow the code to run much faster.

    # First extract the end nodes from each edgelist.  The nodes are numbered so that the start node has number
    # 2*edgenumber-1 and the end node has number 2*edgenumber
    node = np.zeros((Nnodes, 2))
    for n in range(Nedges):
        node[2*n, :] = edgelist[n][0, :]
        node[2*n+1, :] = edgelist[n][-1, :]

    A = np.zeros((Nnodes, Nnodes))
    B = np.zeros((Nedges, Nedges))


    # Now build the adjacency/connection matrices.
    for n in range(Nnodes - 1):
        for m in range(n+1, Nnodes):
            # If nodes m & n are connected
            cond1 = node[n, 0] == node[m, 0]
            cond2 = node[n, 1] == node[m, 1]
            A[n, m] =  cond1 and cond2
            A[m, n] = A[n, m]

            if A[n, m]:
                edgen = int(np.fix(n / 2))
                edgem = int(np.fix(m / 2))
                B[edgen, edgem] = 1
                B[edgem, edgen] = 1

    # If we sum the columns of the adjacency matrix we get the number of other edgelists that are connected to an edgelist
    node_connect = sum(A)              # Connection count array for nodes
    edge_connect = sum(B)              # Connection count array for edges

    # Check every edge to see if any of its ends are connected to just one edge. This should not happen, but
    # occasionally does due to a problem in EDGELINK. Here we simply merge it with the edge it is connected to.
    # Ultimately I want to be able to remove this block of code. I think there are also some cases that are (still)
    # not properly handled by CLEANEDGELIST and there may be a case for repeating this block of code at the end for
    # another final cleanup pass

    for n in range(Nedges):
        if B[(n, n)] == 0 and len(edgelist[n]) != 0:
            [spurdegree, spurnode, startnode, sconns, endnode, econns] = connectioninfo(n, node_connect, edgelist)
            if sconns == 1:
                node2merge = np.where(A[startnode, :])
                [A, B, edgelist, node_connect, edge_connect] = mergenodes(node2merge[0], startnode, A, B, edgelist, node_connect, edge_connect)
            if len(edgelist[n]) != 0:
                if econns == 1:
                    node2merge = np.where(A[endnode, :])
                    [A, B, edgelist, node_connect, edge_connect] = mergenodes(node2merge[0], endnode, A, B, edgelist, node_connect, edge_connect)
    # Now check every edgelist, if the edgelength is below the minimum length check if we should remove it.
    if minlength > 0:
        for n in range(Nedges):
            [spurdegree, spurnode, _, _, _, _] = connectioninfo(n, node_connect, edgelist)
            if len(edgelist[n]) != 0 and edgelistlength(edgelist[n]) < minlength:
                # Remove unconnected lists, or lists that are only connected to themselves.
                if not edge_connect[n] or (edge_connect[n] == 1 and B[n, n] == 1):
                    A, B, edgelist, node_connect, edge_connect = removeedge(n, edgelist, A, B, node_connect, edge_connect)
                elif spurdegree == 2:
                    linkingedges = np.where(B[n, :])
                    if len(linkingedges[0]) == 1:
                        A, B, edgelist, node_connect, edge_connect = removeedge(n, edgelist, A, B, node_connect, edge_connect)
                    else:
                        spurs = n
                        length = edgelistlength(edgelist[n])
                        for i in range(len(linkingedges[0])):
                            [spurdegree, _, _, _, _, _] = connectioninfo(linkingedges[0][i], node_connect, edgelist)
                            if spurdegree:
                                spurs = np.hstack((spurs, linkingedges[0][i]))
                                length = np.hstack((length, edgelistlength(edgelist[linkingedges[0][i]])))
                        linkingedges = np.hstack((linkingedges[0], n))

                        if isinstance(spurs, np.ndarray):
                            i = np.argmin(length)
                            edge2delete = spurs[i]
                        else:
                            i = 0
                            edge2delete = spurs
                        [spurdegree, spurnode, _, _, _, _] = connectioninfo(edge2delete, node_connect, edgelist)
                        node2merge = np.where(A[spurnode, :])

                        if len(node2merge[0]) != 2:
                            raise ValueError('Attempt to merge other than two nodes.')
                        A, B, edgelist, node_connect, edge_connect = removeedge(edge2delete, edgelist, A, B, node_connect, edge_connect)
                        [A, B, edgelist, node_connect, edge_connect] = mergenodes(node2merge[0][0], node2merge[0][1], A, B, edgelist, node_connect, edge_connect)
                elif spurdegree == 3:
                    A, B, edgelist, node_connect, edge_connect = removeedge(n, edgelist, A, B, node_connect, edge_connect)

        for n in range(Nedges):
            if len(edgelist[n]) != 0 and edgelistlength(edgelist[n]) < minlength:
                if not edge_connect[n] or (edge_connect[n] == 1 and B[n, n] == 1):
                    A, B, edgelist, node_connect, edge_connect = removeedge(n, edgelist, A, B, node_connect, edge_connect)
    m = 0
    nedgelist = []
    for n in range(Nedges):
        if len(edgelist[n]) != 0:
            m = m + 1
            nedgelist.append(edgelist[n])
    return nedgelist


def bwmorph(image, operation):
    """
    BWMORPH - Make the extracted edges thin and clean. Refer BWMORPH function in MATLAB.
    ********************************************************************************************************************
    """

    if operation == 'clean':
        lut = mat.repmat(np.vstack((np.zeros((16, 1)), np.ones((16, 1)))), 16, 1)  ## identity
        lut[16, 0] = 0
        bool_lut = lut != 0
        bool_image = image != 0
        image2 = applylut(bool_image, bool_lut);
        return image2

    if operation == 'thin':
        lut1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,1,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
        0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,1,0,0,1,1,0,0])

        lut1 = np.reshape(lut1, (lut1.shape[0], 1))
        bool_lut1 = lut1 != 0

        lut2 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
        0,0,1,1,0,0,0,0,0,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,1,1,0,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,0,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,
        0,1,0,0,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,0,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

        lut2 = np.reshape(lut2, (lut2.shape[0], 1))
        bool_lut2 = lut2 != 0
        bool_image = image != 0

        image2 = bool_image & applylut(applylut(bool_image, bool_lut1), bool_lut2)
        
        return image2


def applylut(bw, lut):
    """
     applylut - Returns the result of a neighbour operation using the lookup table <lut> which can be created by makelut.

    ********************************************************************************************************************
    Usage: [bw2] = applylut(bw, lut)

     Arguments:  bw         - Binary edge image, it is assumed that isolated pixel is eliminated.

                 lut        - Look Up Table. A pre-calculated look up table of neighborhoods pixels.

    ********************************************************************************************************************
    Returns:  bw2 - a binary image that returns pixel in look up table

    """

    if (lut.shape[0], lut.shape[1]) != (512, 1):
        raise ValueError('applylut: LUT size is not as expected <512, 1>.')
    
    nq = np.log2(len(lut))
    n = np.sqrt(nq);

    if np.floor(n) != n:
        raise ValueError("applylut: LUT length is not as expected. Use makelut to create it.")
    power = np.asarray(range(int(nq-1), -1, -1))
    two_power = np.power(2, power)
    w = np.reshape(two_power, [int(n), int(n)])
    w = w.transpose()
    idx = signal.correlate2d(bw, w, mode='same')
#    idx = idx.transpose()
    row, col = idx.shape[0], idx.shape[1]
    idx_re = np.reshape(idx, [1, row * col])
    
    temp = []
    for ind in range(idx_re.shape[1]):
        temp.append(lut[idx_re[0, ind], 0])
    temp = np.asarray(temp)
    temp = np.reshape(temp, [row, col])
    A = temp
    
    return A


def findendsjunctions(edge, disp=0):
    """
     findendsjunctions - find junctions and endings in a line/edge image

    ********************************************************************************************************************
    Arguments:  edgeim - A binary image marking lines/edges in an image.  It is assumed that this is a thinned or
    skeleton image

                disp   - An optional flag 0/1 to indicate whether the edge image should be plotted with the junctions
                and endings marked.  This defaults to 0.

    ********************************************************************************************************************
    Returns:    juncs = [rj, cj] - Row and column coordinates of junction points in the image.

                ends = [re, ce] - Row and column coordinates of end points in the image.

    ********************************************************************************************************************
    """
    # Set up look up table to find junctions.  To do this we use the function defined at the end of this file to test
    # that the centre pixel within a 3x3 neighbourhood is a junction.
    imgtile0 = get_imgtile(edge, start=0)
    imgtile1 = get_imgtile(edge, start=1)
    diff = np.sum(np.abs(imgtile0 - imgtile1), axis=2)
    ends = np.int32(diff == 2) * edge
    junctions = np.int32(diff >= 6) * edge
    juncs = np.where(junctions > 0)
    endcs = np.where(ends > 0)
    return juncs, endcs


def get_imgtile(img, start):
    """
     get_imgtile - reach the neighborhood pixel value of a pixel and store them in depth layer

    ********************************************************************************************************************
    Arguments:  img - a binary image with img.shape=[height, width].

                start - store neighborhood pixel value from <start>.

    ********************************************************************************************************************
    Returns:    imgtile - a 3d numpy array with imgtile.shape=[height, width, 8]. imgtile[hth, wth, :] is the
                neighbourhood pixel value of img[hth, wth].

                for instance:
                if start == 0:
                  imgtile[hth, wth, :] = [t[0], t[1], ..., t[7]]

                    t[0] t[7] t[6]
                    t[1]  o   t[5]
                    t[2] t[3] t[4]

    ********************************************************************************************************************
    """
    height, width = img.shape[0], img.shape[1]

    imgtile = np.zeros((height, width, 8))
    template = np.zeros((height+2, width+2))
    template[1:-1, 1:-1] = img

    ind = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    ind = (ind + start) % 8
    imgtile[:,:,ind[0]] = template[0: height  , 0: width]
    imgtile[:,:,ind[1]] = template[1: height+1, 0: width]
    imgtile[:,:,ind[2]] = template[2: height+2, 0: width]
    imgtile[:,:,ind[3]] = template[2: height+2, 1: width+1]
    imgtile[:,:,ind[4]] = template[2: height+2, 2: width+2]
    imgtile[:,:,ind[5]] = template[1: height+1, 2: width+2]
    imgtile[:,:,ind[6]] = template[0: height+0, 2: width+2]
    imgtile[:,:,ind[7]] = template[0: height+0, 1: width+1]

    return imgtile


def unitvector(v):
    nv = v / np.sqrt(sum(np.square(v)))
    return nv


def availablepixels(rp, cp, edgeNo, edgeim, junct):
    """
    AVAILABLEPIXELS - Find all the pixels that could be linked to point r, c

    ********************************************************************************************************************
    Arguments:  rp, cp - Row, col coordinates of pixel of interest.

                edgeNo - The edge number of the edge we are seeking to track. If not supplied its value defaults to 0
                resulting in all adjacent junctions being returned, (see note below)

    ********************************************************************************************************************
    Returns:    ra, ca - Row and column coordinates of available non-junction pixels.
                rj, cj - Row and column coordinates of available junction pixels.

                A pixel is available for linking if it is:
                1) Adjacent, that is it is 8-connected.
                2) Its value is 1 indicating it has not already been assigned to an edge
                3) or it is a junction that has not been labeled -edgeNo indicating we have
                not already assigned it to the current edge being tracked.  If edgeNo is 0 all adjacent junctions will
                be returned

                If edgeNo not supplied set to 0 to allow all adjacent junctions to be returned
    ********************************************************************************************************************
    """
    if 'edgeNo' not in vars().keys():
        edgeNo = 0


    row, col = edgeim.shape[0], edgeim.shape[1]

    # row and column offsets for the eight neighbours of a point
    roff = np.array([-1, 0, 1, 1, 1, 0, -1, -1])
    coff = np.array([-1, -1, -1, 0, 1, 1, 1, 0])
    r = rp + roff
    c = cp + coff

    # Find indices of arrays of r and c that are within the image bounds
    cond1 = np.where(r >= 0)
    cond2 = np.where(r < row)
    conr = set(cond1[0]) & set(cond2[0])
    cond3 = np.where(c >= 0)
    cond4 = np.where(c < col)
    conc = set(cond3[0]) & set(cond4[0])
    ind = list(conr & conc)

    ra, ca, rj, cj = [], [], [], []
    # A pixel is available for linking if its value is 1 and it is not a labeled junction -edgeNo
    for idx in ind:
        if edgeim[r[idx], c[idx]] == 1 and junct[r[idx], c[idx]] != 1:
            ra.append(r[idx])
            ca.append(c[idx])
        elif edgeim[r[idx], c[idx]] != -edgeNo and junct[r[idx], c[idx]] == 1:
            rj.append(r[idx])
            cj.append(c[idx])

    ra = np.asarray(ra)
    ca = np.asarray(ca)
    rj = np.asarray(rj)
    cj = np.asarray(cj)

    return ra, ca, rj, cj


def drawedgelist(edgelist, lw, col, rowscols, name):
    """
    DRAWEDGELIST - Plot and saved the edgelist
    ********************************************************************************************************************
    """
    if lw == 'Ignore':
        lw = 1
    if col == 'Ignore':
        col = np.array([0, 0, 1])
    Nedge = len(edgelist)
    if rowscols != 'Ignore':
        [rows, cols] = rowscols
    if col == 'rand':
        col = hsv(Nedge)
    elif col == 'mono':
        col = hsv(1)
        col = col * Nedge
    else:
        raise ValueError('Color not specified properly')
        
    plt.figure(figsize=(6,8))


    for idx in range(Nedge):
        plt.plot(edgelist[idx][:, 1], edgelist[idx][:, 0], color=col[idx])
    
    plt.axis([-10, cols + 20, rows + 20, -20])
#    plt.axis('scaled')
    plt.axis('off')
    plt.savefig('%s.jpg' % name, dpi=500, bbox_inches='tight')
    plt.close()
    # plt.show()


def hsv(Nedge):
    # Create random color for <drawedgelist>
    color = []
    for idx in range(Nedge):
        color.append(randomcolor())
    return color


def connectioninfo(n, node_connect, edgelist):
    # Function to provide information about the connections at each end of an edgelist
    # [spurdegree, spurnode, startnode, sconns, endnode, econns] = connectioninfo(n)
    # spurdegree - If this is non-zero it indicates this edgelist is a spur, the value is the number of edges this spur
    # is connected to.
    # spurnode   - If this is a spur spurnode is the index of the node that is connected to other edges, 0 otherwise.
    # startnode  - index of starting node of edgelist.
    # endnode    - index of end node of edgelist.
    # sconns     - number of connections to start node.
    # econns     - number of connections to end node.

    if len(edgelist[n]) == 0:
        spurdegree, spurnode, startnode, sconns, endnode, econns = 0, 0, 0, 0, 0, 0
        return spurdegree, spurnode, startnode, sconns, endnode, econns
    startnode = 2 * n
    endnode = 2 * n + 1
    sconns = node_connect[startnode]
    econns = node_connect[endnode]
    if sconns == 0 and econns >= 1:
        spurdegree = econns
        spurnode = endnode
    elif sconns >= 1 and econns == 0:
        spurdegree = sconns
        spurnode = startnode
    else:
        spurdegree = 0
        spurnode = 0

    return spurdegree, spurnode, startnode, sconns, endnode, econns


def mergenodes(n1, n2, A, B, edgelist, node_connect, edge_connect):
    # Internal function to merge 2 edgelists together at the specified nodes and perform the necessary updates to the
    # edge adjacency and node adjacency matrices and the connection count arrays
    edge1 = int(np.fix(n1 / 2))
    edge2 = int(np.fix(n2 / 2))
    s1 = 2 * edge1
    e1 = 2 * edge1 + 1
    s2 = 2 * edge2
    e2 = 2 * edge2 + 1

    if edge1 == edge2:
        raise TypeError('Try to merge an edge with itself.\n')
    if not A[n1, n2]:
        raise TypeError('Try to merge nodes that are not connected.\n')

    if (n1 % 2):
        flipedge1 = 0
    else:
        flipedge1 = 1

    if n2 % 2:
        flipedge2 = 1
    else:
        flipedge2 = 0

    # Join edgelists together - with appropriate reordering depending on which end is connected to which.  The result
    # is stored in edge1
    if not flipedge1 and not flipedge2:
        edgelist[edge1] = np.vstack((edgelist[edge1], edgelist[edge2]))
        A[e1, :] = A[e2, :]
        A[:, e1] = A[:, e2]
        node_connect[e1] = node_connect[e2]

    elif not flipedge1 and flipedge2:
        edgelist[edge1] = np.vstack((edgelist[edge1], np.flipud(edgelist[edge2])))
        A[e1, :] = A[s2, :]
        A[:, e1] = A[:, s2]
        node_connect[e1] = node_connect[s2]
    elif flipedge1 and not flipedge2:
        edgelist[edge1] = np.vstack((np.flipud(edgelist[edge1]), edgelist[edge2]))
        A[s1, :] = A[e1, :]
        A[:, s1] = A[:, e1]
        A[e1, :] = A[e2, :]
        A[:, e1] = A[:, e2]
        node_connect[s1] = node_connect[e1]
        node_connect[e1] = node_connect[e2]
    elif flipedge1 and  flipedge2:
        edgelist[edge1] = np.vstack((np.flipud(edgelist[edge1]), np.flipud(edgelist[edge2])))
        A[s1, :] = A[e1, :]
        A[:, s1] = A[:, e1]
        A[e1, :] = A[s2, :]
        A[:, e1] = A[:, s2]
        node_connect[s1] = node_connect[e1]
        node_connect[e1] = node_connect[s2]
    else:
        raise ValueError('Edgelists cannot be merged.')

    B[edge1, :] = np.logical_or(B[edge1, :], B[edge2, :]) * 1
    B[:, edge1] = np.logical_or(B[:, edge1], B[:, edge2]) * 1
    B[edge1, edge1] = 0
    edge_connect = sum(B)
    node_connect = sum(A)

    A, B, edgelist, node_connect, edge_connect = removeedge(edge2, edgelist, A, B, node_connect, edge_connect)

    return A, B, edgelist, node_connect, edge_connect


def removeedge(n, edgelist, A, B, node_connect, edge_connect):
    edgelist[n] = []
    edge_connect = edge_connect - B[n, :]
    edge_connect[n] = 0
    B[n, :] = 0
    B[:, n] = 0
    nodes2delete = [2*n, 2*n + 1]
    node_connect = node_connect - A[nodes2delete[0], :]
    node_connect = node_connect - A[nodes2delete[1], :]
    A[nodes2delete, :] = 0
    A[:, nodes2delete] = 0

    return A, B, edgelist, node_connect, edge_connect


def edgelistlength(edgelist):
    diff_sqr = np.square(edgelist[0:-1, :] - edgelist[1:, :])
    row_sqrtsum = np.sqrt(np.sum(diff_sqr, 1))
    l = np.sum(row_sqrtsum)
    return l


def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color


def seglist(edgelist, tol):
    """
    LINESEG - Form straight line segements from an edge list.

    ********************************************************************************************************************
    Usage: seglist = lineseg(edgelist, tol)

    ********************************************************************************************************************

    Arguments:  edgelist - Cell array of edgelists where each edgelist is an Nx2 array of (row col) coords.
                tol      - Maximum deviation from straight line before a segment is broken in two (measured in pixels).

                Returns:
                seglist  - A cell array of in the same format of the input edgelist but each seglist is a subsampling
                of its corresponding edgelist such that straight line segments between these subsampled points do not
                deviate from the original points by more than tol.

    ********************************************************************************************************************
    This function takes each array of edgepoints in edgelist, finds the size and position of the maximum deviation from
    the line that joins the endpoints, if the maximum deviation exceeds the allowable tolerance the edge is shortened to
    the point of maximum deviation and the test is repeated. In this manner each edge is broken down to line segments,
    each of which adhere to the original data with the specified tolerance.

    ********************************************************************************************************************
    """
    Nedge = len(edgelist)
    seglist = []
    
    for e in range(Nedge):
        temp = []
        # Note that (col, row) corresponds to (x,y)
        y = edgelist[e][:, 0]
        y = np.reshape(y, (1, y.shape[0]))
        x = edgelist[e][:, 1]
        x = np.reshape(x, (1, x.shape[0]))
        fst = 0             # Indices of first and last points in edge
        lst = x.shape[1] - 1        # Segment being considered
        Npts = 1
        temp.append(np.asarray([y[0, fst], x[0, fst]]))

        while fst < lst:
            [m, i, D, s] = maxlinedev(x[0, fst:lst+1], y[0, fst:])       # Find size and position of maximum deviation
            tol1 = tol * ((10 - np.exp(-s)) / (10 + np.exp(-s)))
            while m > tol1:      # while deviation is > tol
                lst = i + fst    # Shorten line to point of max deviation by adjusting lst
                [m, i, D, s] = maxlinedev(x[0, fst:lst+1], y[0, fst:lst+1])
                tol1 = tol * ((10 - np.exp(-s)) / (10 + np.exp(-s)))
            Npts = Npts + 1
            temp.append(np.asarray([y[0, lst], x[0, lst]]))
            fst = lst
            lst = x.shape[1] - 1
        temp = np.asarray(temp)
        seglist.append(temp)
    return seglist


def maxlinedev(x, y):
    """
    MAXLINEDEV - Find max deviation from a line in an edge contour.

    ********************************************************************************************************************
    Function finds the point of maximum deviation from a line joining the endpoints of an edge contour.

    ********************************************************************************************************************
    Usage:   [maxdev, index, D, totaldev] = maxlinedev(x,y)

    ********************************************************************************************************************
    Arguments:
        x, y   - arrays of x,y  (col,row) indicies of connected pixels on the contour.

    ********************************************************************************************************************
    Returns:
        maxdev   - Maximum deviation of contour point from the line joining the end points of the contour (pixels).
        index    - Index of the point having maxdev.
        D        - Distance between end points of the contour so that one can calculate maxdev/D - the normalised error.
        totaldev - Sum of the distances of all the pixels from the line joining the endpoints.

    ********************************************************************************************************************
    """
    Npts = len(x)
    if Npts == 1:
        raise ValueError('Contour of length 1')
        maxdev, index, D, totaldev = 0, 0, 1, 0
        return maxdev, index, D, totaldev
    elif Npts == 0:
        raise ValueError('Contour of length 0')
    x_diff_sq = np.square(x[0] - x[-1])
    y_diff_sq = np.square(y[0] - y[-1])
    D = np.sqrt(x_diff_sq + y_diff_sq)

    if D > 2.2204e-16:
        y1my2 = y[0] - y[-1]
        x2mx1 = x[-1] - x[0]
        C = y[-1] * x[0] - y[0] * x[-1]

        # Calculate distance from line segment for each contour point
        d = abs(x * y1my2 + y * x2mx1 + C) / D
    else:
        # End points are coincident, calculate distances from 1st point
        x_sq = np.square(x - x[0])
        y_sq = np.square(y - y[0])
        D = 1
        d = np.sqrt(x_sq + y_sq)
    [maxdev, index] = [np.max(d), np.argmax(d)]
    totaldev = sum(np.square(d))
    return maxdev, index, D, totaldev


def clean_dir(outpath):
    # clean_path = os.path.join(path, dirname)

    if os.path.isdir(outpath):
        shutil.rmtree(outpath)
        print('Clean directory %s\n' % outpath)
    else:
        print('No directory\n')


def create_dir(outpath):
    # cre_path = os.path.join(path, dirname)
    isexists = os.path.exists(outpath)

    if not isexists:
        os.makedirs(outpath)

        print('Created directory %s\n' % outpath)
    else:
        print('Directory existed\n')














