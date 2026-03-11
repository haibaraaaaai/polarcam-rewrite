Reconstruct phi and theta base on fourkas formula, on polarization signal collected from back reflection of circularly polarized laser on a gold nanorod, into four separate APDs each for one of the polarization channels, so there's no position information unlike with polarization camera.

alpha = np.arcsin(NA / n)
and 3 constants derived from Fourkas formula:
    A  = 1/6  - ca/4        + (ca**3)/12
    B  =        ca/8        - (ca**3)/8
    C  = 7/48 - ca/16  - (ca**2)/16 - (ca**3)/48

anisotropy_x = (c0-c90)/(c0+c90)
anisotropy_y = (c45-c135)/(c45+c135)

and r = sqrt(anis_x^2 + anis_y^2)

phi = 0.5 * np.arctan2(anis_y, anis_x) which is pi degenerate:
phi      = np.unwrap(phi, period=np.pi)

And theta from:
r = C*sin²θ/(A+B*sin²θ)
which only goes from 0-90 degrees.

There's mainly three corrections:
1. channel correction, there's a T-Icor_Matrix base on measured optics from our APD setup, and a, b factors, "a" from the fact that Fourkas demands C0+C90 = C45+C135 (not demand as in if that's not true fourkas does not work, but base on fourkas construct if we have polarization signal perfectly collected C0+C90 = C45+C135 is just true), and "b" from base noise thing, only used to make sure the signal from all time is non-negative. 

2. ansitropy centering, this is slightly arbitrary, we use it to correct for the sample not centered at laser focus / and background correction. It's base on the believe we selected good "helicopters", spinners that have constant theta and changing phi, so because phi is related to the orientation of r, and theta the length of r, we expect the anistropy plot (x, y scatter but you can maybe also just plot r vector) to be a perfect circle centered at (0,0). We do want maybe a better centering method, base on the fact that to get through all the phi angles, (needed to make a full rotation), r need to spin around twice, but both time (0,0) should be within the trajectory formed by r. (the helicoppter case is becasue both half of phi (0-180 and 180-360) makes  teh same circular trajectory on anis space and they perfectly overlap)

3. linearity correction. This is to correct for our APD setup optics, where we think if we remove the sample and pass directly a polarized light, if we change the polarization angle continuously and uniformly, due to optics distortion the result we recovered will not be uniform. So we made the correction basically by setting the spacing between angles uniform. But this assumes the sample has no internal symmetry / preferred angles, and often is just not true or at least can not be assumed. And we made some rough measurement to show linearity is not that bad anyway. so it's default off.

there is one extra correction attempted here, in teh first part of the notebook:
The idea is that given alpha base on NA and n, there's a limit of max r allowed, base on formula, when setting sin²θ = 1, r_max = C / (A+B) and so any r beyond that value gives unphysically theta.
Background and many other factors affect r, we should correct for background specifically somewhere also, but the point is that background, when you add to say anis_x, (c0+b0-c90-b90)/(c0+b0+c90+b90), actually shrinks the anis values thus make r smaller. 

So when we are getting larger r than allowed, it's because we need to estimate and account for optics' effect on NA. So we fit an effective NA base on distribution of r from samples we know some should hit / go over theta = 90. It's not the most rock solid method, but it's a good quick fix.
