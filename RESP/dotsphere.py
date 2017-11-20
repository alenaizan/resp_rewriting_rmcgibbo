import numpy as np

def rad(A):
    """
    Convert an angle from degrees to radians
    """
    return A*0.017453293

dp_tol = 0.001
n_refine_steps = 25
r_h = np.sqrt(1.0 - 2.0*np.cos(rad(72)))/(1.0 - np.cos(rad(72)))


def divarc(xyz1, xyz2, div1, div2):
    """
    Divide an arc based on the great circle
    """
    xd = xyz1[1] * xyz2[2] - xyz2[1] * xyz1[2]
    yd = xyz1[2] * xyz2[0] - xyz2[2] * xyz1[0]
    zd = xyz1[0] * xyz2[1] - xyz2[0] * xyz1[1]
    dd = np.linalg.norm([xd, yd, zd])
    if dd < dp_tol:
        raise Exception('_divarc: rotation axis of length')

    d1 = np.linalg.norm(xyz1)
    if d1 < np.sqrt(0.5):
        raise Exception('_divarc: vector 1 of sq.length too small')

    d2 = np.linalg.norm(xyz2)
    if d2 < np.sqrt(0.5):
        raise Exception('_divarc: vector 2 of sq.length too small')

    phi = np.sin(dd / np.sqrt(d1*d2))
    phi = phi * div1/div2
    sphi = np.sin(phi)
    cphi = np.cos(phi)
    s = (xyz1[0]*xd + xyz1[0]*yd + xyz1[2]*zd) / dd
    x = xd*s*(1.0-cphi)/dd + xyz1[0] * cphi + (yd*xyz1[2] - xyz1[1]*zd)*sphi/dd
    y = yd*s*(1.0-cphi)/dd + xyz1[1] * cphi + (zd*xyz1[0] - xyz1[2]*xd)*sphi/dd
    z = zd*s*(1.0-cphi)/dd + xyz1[2] * cphi + (xd*xyz1[1] - xyz1[0]*yd)*sphi/dd
    dd = np.linalg.norm([x, y, z])

    return np.array([x/dd, y/dd, z/dd])


def icosahedron_vertices():
    """
    Compute the vertices of an icosahedron.
    Return: 
        verts: np.ndarray, shape=(12, 3)
               Caertesian coordinates of the 12 verticles of a unit icosahedron.
    """
    rg = np.cos(rad(72))/(1-np.cos(rad(72)))
    verts = []
    verts.append([0.0, 0.0, 1.0])
    angles = [72, 144, 216, 288]
    for a in angles:
        verts.append([r_h*np.cos(rad(a)), r_h*np.sin(rad(a)), rg])
    verts.append([r_h, 0, rg])
    angles = [36, 108]
    for a in angles:
        verts.append([r_h*np.cos(rad(a)), r_h*np.sin(rad(a)), -rg])
    verts.append([-r_h, 0, -rg])
    angles = [252, 324]
    for a in angles:
        verts.append([r_h*np.cos(rad(a)), r_h*np.sin(rad(a)), -rg])
    verts.append([0, 0, -1])

    return np.array(verts)


def dotsphere1(density):
    """
    Create a dot distribution over the unit shpere based on repeated
    splitting and refining the arcs of an icosahedron.

    Parameters
    ----------
    density : int
         Required number of dots on the unit sphere

    Returns
    -------
    dots : np.ndarray, shape=(N, 3), dtype=np.double
         Dots on the surface of the unit sphere. The number of dots will be
         at minimum equal to the `density` argument, but will be roughly two
         times larger.

    See Also
    --------
    dotsphere2 : acomplished the same goal, but based on splitting
         the faces. The two procedures are capable of yielding different
         number of points because of the different algorithms used.
    """

    # calculate tessalation level
    a = np.sqrt((density-2.0)/10.0)
    tess = int(np.ceil(a))
    vertices = icosahedron_vertices()

    if tess > 1:
        a = r_h*r_h*2.0*(1.0 - np.cos(rad(72.0)))
        # Calculate tessalation of icosahedron edges
        for i in range(11):
            for j in range(i+1, 12):
                d = np.linalg.norm(vertices[i] - vertices[j])
                if abs(a-d**2) > dp_tol:
                    continue
                for tl in range(tess):
                    vertices = np.concatenate((vertices, divarc(vertices[i], vertices[j], tl, tess).reshape(1,3)))
    
    # Calculate tessalation of icosahedron faces
    for i in range(10):
        for j in range(i+1, 11):
            d = np.linalg.norm(vertices[i]- vertices[j])
            if abs(a-d**2) > dp_tol:
                continue

            for k in range(j+1, 12):
                d_ik = np.linalg.norm(vertices[i] - vertices[k])
                d_jk = np.linalg.norm(vertices[j] - vertices[k])
                if (abs(a-d_ik**2) > dp_tol) or (abs(a-d_jk**2) > dp_tol):
                    continue
                for tl in range(1, tess-1):
                    ji = divarc(vertices[j], vertices[i], tl, tess)
                    ki = divarc(vertices[k], vertices[i], tl, tess)

                    for tl2 in range(1, tess-tl):
                        ij = divarc(vertices[i], vertices[j], tl2, tess)
                        kj = divarc(vertices[k], vertices[j], tl2, tess)
                        ik = divarc(vertices[i], vertices[k], tess-tl-tl2, tess)
                        jk = divarc(vertices[j], vertices[k], tess-tl-tl2, tess)

                        xyz1 = divarc(ki, ji, tl2, tess-tl)
                        xyz2 = divarc(kj, ij, tl, tess-tl2)
                        xyz3 = divarc(jk, ik, tl, tl+tl2)

                        x = xyz1 + xyz2 + xyz3
                        vertices = np.concatenate((vertices, (x / np.linalg.norm(x)).reshape(1, 3)))
    return vertices

def dotsphere2(density):
    """
    Create a dot distribution over the unit shpere based on repeated
    truncating and refining the faces of an icosahedron.

    Parameters
    ----------
    density : int
        Required number of dots on the unit sphere

    Returns
    -------
    dots : np.ndarray, shape=(N, 3), dtype=np.double
        Dots on the surface of the unit sphere. The number of dots will be
        at minimum equal to the `density` argument, but will be roughly two
        times larger.

    See Also
    --------
    dotsphere_icos1 : acomplished the same goal, but based on splitting
        the edges. The two procedures are capable of yielding different
        number of points because of the different algorithms used.
    """
    a = np.sqrt((density - 2.0)/ 30.0)
    tess = max([int(np.ceil(a)), 1])
    vertices = icosahedron_vertices()

    a = r_h * r_h * 2.0 * (1.0 - np.cos(rad(72.0)))

    # Dodecaeder vertices
    for i in range(10):
        for j in range(i+1, 11):
            d = np.linalg.norm(vertices[i] - vertices[j])
            if abs(a-d*d) > dp_tol:
                continue
            for k in range(j+1, 12):
                d_ik = np.linalg.norm(vertices[i] - vertices[k])
                d_jk = np.linalg.norm(vertices[j] - vertices[k])
                if (abs(a - d_ik**2) > dp_tol) or (abs(a - d_jk**2) > dp_tol):
                    continue
                x = vertices[i] + vertices[j] + vertices[k]
                vertices = np.concatenate((vertices, (x / np.linalg.norm(x)).reshape(1,3)))
    if tess > 1:
        # square of the edge of an dodecaeder
        adod = 4.0 * (np.cos(rad(108.)) - np.cos(rad(120.))) / (1.0 - np.cos(rad(120)))
        # square of the distance of two adjacent vertices of ico- and dodecaeder
        ai_d = 2.0 * (1.0 - np.sqrt(1.0 - a/3.0))
        # calculate tessalation of mixed edges
        for i in range(31):
            j1 = 12
            j2 = 32
            a = ai_d
            if i > 12:
                j1 = i + 1
                a = adod
            for j in range(j1, j2):
                d = np.linalg.norm(vertices[i] - vertices[j])
                if abs(a-d*d) > dp_tol:
                    continue
                for tl in range(1, tess):
                    vertices = np.concatenate((vertices, divarc(vertices[i], vertices[j], tl, tess).reshape(1,3)))

        # calculate tessalation of pentakisdodecahedron faces
        for i in range(12):
            for j in range(12, 31):
                d = np.linalg.norm(vertices[i] - vertices[j])
                if abs(ai_d - d*d) > dp_tol:
                    continue
                for k in range(j + 1, 32):
                    d_ik = np.linalg.norm(vertices[i] - vertices[k])
                    d_jk = np.linalg.norm(vertices[j] - vertices[k])
                    if (abs(ai_d - d_ik**2) > dp_tol) or (abs(adod - d_jk**2) > dp_tol):
                        continue
                    for tl in range(1, tess-1):
                        ji = divarc(vertices[j], vertices[i], tl, tess)
                        ki = divarc(vertices[k], vertices[i], tl, tess)
                        for tl2 in range(1, tess - tl):
                            ij = divarc(vertices[i], vertices[j], tl2, tess)
                            kj = divarc(vertices[k], vertices[j], tl2, tess)
                            ik = divarc(vertices[i], vertices[k], tess-tl-tl2, tess)
                            jk = divarc(vertices[j], vertices[k], tess-tl-tl2, tess)

                            xyz1 = divarc(ki, ji, tl2, tess-tl)
                            xyz2 = divarc(kj, ij, tl, tess-tl2)
                            xyz3 = divarc(jk, ik, tl, tl+tl2)

                            x = xyz1 + xyz2 + xyz3
                            vertices = np.concatenate((vertices, (x / np.linalg.norm(x)).reshape(1,3)))

    return vertices

#****************************************************************************
# Code for refining a dot distribution based on electrostatic repulsion
#****************************************************************************

def get_coulomb_energy(points):
    # Calculate the coulomb energy between a set of points
    e = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
                e += 1.0/(np.linalg.norm(points[i] - points[j]))
    return e

def get_coulomb_forces(points):
    forces = np.zeros(points.shape)

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            r = points[i] - points[j]
            l = np.linalg.norm(r)
            if l < 1e-8:
                continue
            ff = r / l**3
            forces[i] += ff
            forces[j] -= ff
    return forces

def refine_dotsphere(vertices):
    """
    Refine a dot distribution over the unit sphere using an iterative
    electrostatic repulsion-type approach.

    Parameters
    ----------
    vertices : np.ndarray, dtype=np.double
        The initial vertices

    Returns
    -------
    vertices: Updated vertices
    """

    step = 0.005
    if len(vertices) > 100:
        step /=50.0
    e0 = get_coulomb_energy(vertices)
    for i in range(n_refine_steps):
        forces = get_coulomb_forces(vertices)
        for j in range(len(vertices)):
            vertices[j] += forces[j] * step
            vertices[j] /= np.linalg.norm(vertices[j])
        e = get_coulomb_energy(vertices)
        if e0 < e:
            step /= 2.0
        e0 = e
        if step < 1e-8:
            break
    return vertices

def dotsphere(density):
    """
    Create a dot distribution over the unit sphere, choosing the most
    appropriate implementation based on the number of dots you request.

    Parameters
    ----------
    density : int
        Required number of dots on the unit sphere

    Returns
    -------
    vertices : np.ndarray, ndtype=np.double
    """
    density = int(density)
    i1 = 1
    i2 = 1
    while 10*i1*i1 + 2 < density:
        i1 += 1
    while 30*i2*i2 + 2 < density:
        i2 += 1

    # Use one of the two algorithms to generate the initial dots
    # they will give slightly too many.
    if 10*i1*i1-2 < 30*i2*i2-2:
        vertices = dotsphere1(density)
    else:
        vertices = dotsphere2(density)
    keep = np.zeros(density, dtype=int)
    if density < len(vertices):
        import random
        random.seed(0)
        for i in range(len(keep)):
            keep[i] = random.randint(0, len(vertices)-1)
    vertices = refine_dotsphere(vertices[keep])
    return vertices
