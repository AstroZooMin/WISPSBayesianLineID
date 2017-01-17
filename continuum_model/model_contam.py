import os
import numpy as np
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
import astropy.io.fits as fitsio
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

pixfrac_corr = 0.12825 / 0.08
zp = {'F110':26.8223,'F160':25.9463}

def inEllipse(x, y, xc, yc, a, b, theta):
    """
    (x,y) is coordinate of point being tested
    the region's (x,y) is center of ellipse
    the region's a and b are semimajor and semiminor axes of ellipse
    the region's theta is rotation angle of ellipse in degrees, ccw from x-axis
    returns Boolean
    """
    thetaRad = theta*np.pi/180
    num1 = np.cos(thetaRad)*(x-xc) + np.sin(thetaRad)*(y-yc)
    num2 = np.sin(thetaRad)*(x-xc) - np.cos(thetaRad)*(y-yc)
    return (np.power(num1,2)/np.power(a,2)) + (np.power(num2,2)/np.power(b,2)) <= 1

def bboxEllipse(a,b,theta):
    """
    the region's (x,y) is center of ellipse
    the region's a and b are semimajor and semiminor axes of ellipse
    the region's theta is rotation angle of ellipse in degrees, ccw from x-axis
    returns x, y, dx, dy
    """
    thetaRad = theta*np.pi/180
    dx = np.sqrt( pow(b*np.sin(thetaRad), 2) + pow(a*np.cos(thetaRad), 2) )
    dy = np.sqrt( pow(a*np.sin(thetaRad), 2) + pow(b*np.cos(thetaRad), 2) )
    return dx, dy

def get_waves(x,hdr):
    return (x - hdr['CRPIX1']) * hdr['CDELT1'] + hdr['CRVAL1']

def get_template(tmpl_id,tmpl_norm,tmpl_dir,interpolate=True):

    tmpl_name = os.path.join(tmpl_dir,tmpl_id)
    template = np.genfromtxt(tmpl_name,dtype=[('waves',float),('flux',float)])
    if interpolate:
        interp = scipy.interpolate.interp1d(template['waves'],template['flux'])
        waves = np.arange(template['waves'][0],template['waves'][-1],1)
        flux  = interp(waves)
        template = np.array(zip(waves,flux),dtype=[('waves',float),('flux',float)])
    template['flux'] /= tmpl_norm
    return template

def get_response(grism):
    response = fitsio.getdata('config/WFC3.IR.%s.1st.sens.2.fits'%grism)
    return scipy.interpolate.interp1d(response['WAVELENGTH'],response['SENSITIVITY'],bounds_error=False,fill_value=0)

def convolve_tmpl_resp(template,z,response):
    template['waves'] *= (1+z)
    template['flux'] *= response(template['waves'])
    return template

def get_flux(w0,w1,template):
    cond = (w0<=template['waves']) & (template['waves']<w1)
    if len(template[cond]) == 0: return 0
    flux = scipy.integrate.simps(template['flux'][cond],x=template['waves'][cond]) / (w1-w0)
    return flux

def get_direct_catalog(direct_catname,img_hdr):

    _catalog = np.genfromtxt(direct_catname,
                            dtype=[('RA_DEC_NAME','>S25'),('OBJ_NUM',int),('X_IMAGE',float),('Y_IMAGE',float),
                                   ('A_IMAGE',float),('B_IMAGE',float),('THETA_IMAGE',float),('X_WORLD',float),
                                   ('Y_WORLD',float),('A_WORLD',float),('B_WORLD',float),('THETA_WORLD',float),
                                   ('MAG',float),('MAGERR_AUTO',float),('CLASS_STAR',float),('FLAGS',int)])

    img_wcs = WCS(img_hdr)
    _catalog['X_IMAGE'],_catalog['Y_IMAGE'] = img_wcs.all_world2pix(_catalog['X_WORLD'],_catalog['Y_WORLD'],1)
    _catalog['A_IMAGE'] *= pixfrac_corr
    _catalog['B_IMAGE'] *= pixfrac_corr

    bbox_x, bbox_y = bboxEllipse(_catalog['A_IMAGE'],_catalog['B_IMAGE'],_catalog['THETA_IMAGE'])

    catalog = np.recarray(len(_catalog),dtype=_catalog.dtype.descr+[('BBOX_X',float),('BBOX_Y',float)])
    for x in _catalog.dtype.names: catalog[x] = _catalog[x]

    catalog['BBOX_X']  = 2*bbox_x
    catalog['BBOX_Y']  = 2*bbox_y
    return catalog

def get_direct_contams(catalog,entry,dx,dy):
    xc, yc = entry['X_IMAGE'],entry['Y_IMAGE']
    cond1 = (catalog['X_IMAGE']-xc <=  dx+catalog['BBOX_X']*3)
    cond2 = (catalog['X_IMAGE']-xc >= -dx-catalog['BBOX_X']*3)
    cond3 = (catalog['Y_IMAGE']-yc <=  dy+catalog['BBOX_Y']*3)
    cond4 = (catalog['Y_IMAGE']-yc >= -dy-catalog['BBOX_Y']*3)
    cond  = cond1 & cond2 & cond3 & cond4
    cond[catalog['OBJ_NUM'] == entry['OBJ_NUM']] = False
    contams = catalog[cond]
    return contams

def mask_direct_image(img,sources,fill_value):
    masked = img.copy()
    for source in sources:
        x, y = source['X_IMAGE']-1, source['Y_IMAGE']-1
        dx, dy = 3*source['BBOX_X'], 3*source['BBOX_Y']
        xmin = int(np.floor(max(0, x-dx)))
        xmax = int(np.ceil(min(img.shape[1], x+dx)))
        ymin = int(np.floor(max(0, y-dy)))
        ymax = int(np.ceil(min(img.shape[0], y+dy)))
        gx,gy = np.meshgrid(range(xmin,xmax),range(ymin,ymax))
        ix,iy = gx.flatten(), gy.flatten()
        cond = inEllipse(ix,iy,source['X_IMAGE'],source['Y_IMAGE'],3.5*source['A_IMAGE'],3.5*source['B_IMAGE'],source['THETA_IMAGE'])
        cond = np.array(cond, dtype=bool)
        ix,iy = ix[cond], iy[cond]
        ix,iy = np.array(ix,dtype=int),np.array(iy,dtype=int)
        masked[iy,ix] = fill_value
    return masked

def get_stamp(img,entry,cutx,cuty,fill_value=0,mask=True):

    if mask:

        img = img.copy()
        x, y = entry['X_IMAGE']-1, entry['Y_IMAGE']-1
        dx, dy = 100*entry['BBOX_X'], 100*entry['BBOX_Y']
        xmin = int(np.floor(max(0, x-dx)))
        xmax = int(np.ceil(min(img.shape[1], x+dx)))
        ymin = int(np.floor(max(0, y-dy)))
        ymax = int(np.ceil(min(img.shape[0], y+dy)))
        gx,gy = np.meshgrid(range(xmin,xmax),range(ymin,ymax))
        ix,iy = gx.flatten(), gy.flatten()
        cond = inEllipse(ix,iy,entry['X_IMAGE'],entry['Y_IMAGE'],entry['A_IMAGE'],entry['B_IMAGE'],entry['THETA_IMAGE'])
        cond = np.array(~cond, dtype=bool)
        ix,iy = ix[cond], iy[cond]
        ix,iy = np.array(ix,dtype=int),np.array(iy,dtype=int)
        img[iy,ix] = fill_value

    pos    = (entry['X_IMAGE']-1, entry['Y_IMAGE']-1)
    size   = (cuty,cutx)
    stamp  = Cutout2D(data=img,position=pos,size=size,mode='partial',fill_value=0).data
    stamp = np.ma.masked_array(stamp,mask=~np.isfinite(stamp),fill_value=np.NaN)
    return stamp

def mk_contam(par,entry,grism,direct,tmpl_dir,output_dir='.',savefits=False):

    objid = entry['target']
    z = entry['redshift']
    tmpl_id = entry['model']
    tmpl_norm = entry['Normalization']

    response = get_response(grism)
    template = get_template(tmpl_id,tmpl_norm,tmpl_dir=tmpl_dir)
    template = convolve_tmpl_resp(template,z,response)
    
    grism_name = os.path.join(parent_dir,"Par%i"%par,"%s_DRIZZLE"%grism,"aXeWFC3_%s_mef_ID%i.fits"%(grism,objid))
    grism_img,grism_hdr = fitsio.getdata(grism_name,extname='SCI',header=True)
    grism_wht   = fitsio.getdata(grism_name,extname='WHT')
    grism_waves = get_waves(np.arange(grism_img.shape[1])+2,grism_hdr)
    print grism_waves
    assert len(np.unique(np.diff(grism_waves))) == 1
    grism_dwaves = np.unique(np.diff(grism_waves))[0]
    if grism_img.shape[0]%2!=0: grism_img = grism_img[:-1,:]
    dy = dx = grism_img.shape[0]
    #if dy%2!=0: dx = dy = dy-1

    direct_img,direct_img_hdr = fitsio.getdata(os.path.join(parent_dir,"Par%i"%par,"DATA/DIRECT_GRISM","%s.fits"%direct),extname='SCI',header=True)
    direct_img     =  direct_img - direct_img_hdr['MDRIZSKY']
    direct_catname = os.path.join(parent_dir,"Par%i"%par,"DATA/DIRECT_GRISM","fin_%s.cat"%direct)
    direct_cat     = get_direct_catalog(direct_catname,direct_img_hdr)
    direct_entry   = direct_cat[direct_cat['OBJ_NUM'] == objid][0]
    #direct_contams = get_direct_contams(direct_cat,direct_entry,dx=dx,dy=dy)
    #direct_masked  = mask_direct_image(direct_img,direct_contams,fill_value=0)
    direct_stamp   = get_stamp(direct_img,direct_entry,cutx=dx,cuty=dy)
    
    # For plotting
    direct_stamp_msk   = direct_stamp.copy()
    direct_stamp_unmsk = get_stamp(direct_img,direct_entry,cutx=dx,cuty=dy,mask=False)
    norm = 10**((direct_entry['MAG'] - zp[direct]) / -2.5)
    direct_stamp /= norm

    spex_mod = fitsio.getdata(grism_name,extname='MOD')
    # spex_mod1d = np.average(spex_mod,axis=1)
    # spex_mod2d = np.zeros(direct_stamp.shape)
    # for i in range(direct_stamp.shape[1]): spex_mod2d[:,i] = spex_mod1d
    # norm2 = np.sum(direct_stamp * spex_mod2d)
    # direct_stamp = direct_stamp * spex_mod2d / norm2

    model = np.zeros(grism_img.shape)

    for i,w in enumerate(grism_waves):

        w0,w1 = w-grism_dwaves, w+grism_dwaves
        ampl = get_flux(w0,w1,template)
        ymin,ymax = 0,dy
        xmin,xmax = i-dx/2,i+dx/2
        if xmin != max(0,xmin):
            galaxy = direct_stamp[:,-xmin:]
            xmin = max(0,xmin)
        elif xmax != min(grism_img.shape[1]-1,xmax):
            galaxy = direct_stamp[:,:dx-(xmax-grism_img.shape[1]+1)]
            xmax = min(grism_img.shape[1]-1,xmax)
        else:
            galaxy = direct_stamp

        spex_mod1d = spex_mod[:,i]
        norm = np.ma.sum(np.ma.sum(direct_stamp,axis=1) * spex_mod1d)
        spex_mod2d = np.zeros(galaxy.shape)
        for i in range(galaxy.shape[1]): spex_mod2d[:,i] = spex_mod1d
        model[ymin:ymax,xmin:xmax] += (ampl * galaxy * spex_mod2d / norm)
    
    """
    plt.plot(grism_waves,np.sum(grism_img*spex_mod,axis=0),c='b')
    plt.plot(grism_waves,np.sum(model,axis=0)*np.sum(spex_mod,axis=0),c='r')
    plt.plot(template['waves'],template['flux'],c='k')
    plt.show()
    """

    clean = grism_img - model

    if savefits:
        clean_name = os.path.join(output_dir,"fits/Par%i_ID%i_%s_contam_clean.fits"%(par,objid,grism))
        clean_hdr  = grism_hdr.copy()
        clean_hdr['OBJECT'] += "[continuum subtracted]"
        fitsio.writeto(clean_name,clean,header=clean_hdr,clobber=True)

    return direct_stamp_msk,direct_stamp_unmsk,grism_img,grism_hdr,model,clean

def mk_plot(stamp,_stamp,grism_img,grism_hdr,model,clean,savename):

    fig,axes = plt.subplots(3,1,figsize=(10,12),dpi=75,sharex=True)
    fig.subplots_adjust(left=0.05,right=0.95,bottom=0.13,top=0.78,hspace=0,wspace=0)
    for ax in axes: ax.yaxis.set_visible(False)

    c,r = np.indices(grism_img.shape)
    c = c - grism_img.shape[0]/2.
    r = get_waves(r,grism_hdr)

    sig = np.std(grism_img)
    sig = np.std(np.clip(grism_img,-5*sig,5*sig))
    vmin,vmax = -2.5*sig, +2.5*sig

    axes[0].pcolormesh(r,c,grism_img, vmin=vmin,vmax=vmax,cmap=plt.cm.Greys_r)
    axes[1].pcolormesh(r,c,model,     vmin=vmin,vmax=vmax,cmap=plt.cm.Greys_r)
    im = axes[2].pcolormesh(r,c,clean,vmin=vmin,vmax=vmax,cmap=plt.cm.Greys_r)
    axes[2].set_xlabel('Observed Wavelength [$\AA$]')
    cbaxes = fig.add_axes([0.05, 0.05, 0.9, 0.02])
    cbax   = fig.colorbar(mappable=im, cax=cbaxes, orientation='horizontal')

    dax1 = fig.add_axes([0.25,0.8,0.15,0.15])
    dax2 = fig.add_axes([0.60,0.8,0.15,0.15])
    for dax in [dax1,dax2]:
        dax.set_aspect(1.)
        dax.xaxis.set_visible(False)
        dax.yaxis.set_visible(False)

    sig = np.std(_stamp)
    sig = np.std(np.clip(_stamp,-5*sig,+5*sig))
    vmin,vmax = -3*sig,3*sig

    dax1.imshow(stamp ,interpolation='none',origin='lower',cmap=plt.cm.Greys_r,vmin=vmin,vmax=vmax)
    dax2.imshow(_stamp,interpolation='none',origin='lower',cmap=plt.cm.Greys_r,vmin=vmin,vmax=vmax)

    if savename:
        fig.savefig(savename)
        plt.close(fig)

if __name__ == '__main__':
    
    parent_dir = "/data/highzgal/PUBLICACCESS/WISPS/data/V5.0/"
    cwd = "/data/highzgal/mehta/WISP/WISPSBayesianLineID/"
    tmpl_dir = os.path.join(cwd,"bc03_models")
    conf_dir = os.path.join(cwd,"continuum_model/config")
    redshifts = fitsio.getdata(os.path.join(cwd,'Par96_estimates.fits'))

    # for entry in redshifts[:100]:
    #     print "Obj#%i"%entry['target']
    #     for grism,filt in zip(['G102','G141'],['F110','F160']):
    #         stamp,_stamp,grism_img,grism_hdr,model,clean = mk_contam(par=96,entry=entry,grism=grism,direct=filt,tmpl_dir=tmpl_dir,savefits=False)
    #         mk_plot(stamp,_stamp,grism_img,grism_hdr,model,clean,savename='plots/Par96_ID%i_%s.png' % (entry['target'],grism))
    
    entry = redshifts[redshifts['target']==13][0]
    print entry
    stamp,_stamp,grism_img,grism_hdr,model,clean = mk_contam(par=96,entry=entry,grism='G141',direct='F160',tmpl_dir=tmpl_dir,savefits=False)
    mk_plot(stamp,_stamp,grism_img,grism_hdr,model,clean,savename=None)
    plt.show()