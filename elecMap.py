# import numpy as np
# import matplotlib.pyplot as plt
# import os

try:
    import tkinter as tk
except ImportError as e:
    raise ImportError("Tkinter is not available in this Python environment. Please ensure you have Tkinter installed and properly configured. If you are using a virtual environment, you might need to reinstall Python with Tkinter support or activate a different environment that has Tkinter.") from e
    
import os
import tkinter as tk
import scipy.io
from PIL import Image, ImageTk

class ImageViewerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Viewer")

        self.current_index = 0
        self.mat_files = []
        self.directory_path = 'mat_api'  # Provide the directory path here
        self.load_mat_files()
        
        self.prev_button = tk.Button(self.master, text="Previous", command=self.show_previous)
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(self.master, text="Next", command=self.show_next)
        self.next_button.pack(side=tk.RIGHT)

        self.image_label = tk.Label(self.master)
        self.image_label.pack()

        self.load_image()

    def load_mat_files(self):
        self.mat_files = [os.path.join(self.directory_path, f) for f in os.listdir(self.directory_path) if f.endswith('.mat')]

    def load_image(self):
        mat_file = self.mat_files[self.current_index]
        image_data = scipy.io.loadmat(mat_file)['image_data']
        image = Image.fromarray(image_data)
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep a reference to prevent garbage collection

    def show_next(self):
        if self.current_index < len(self.mat_files) - 1:
            self.current_index += 1
            self.load_image()

    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

def main():
    root = tk.Tk()
    app = ImageViewerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()




    def load_files(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("MAT files", "*.mat")])
        self.file_paths.extend(file_paths)

    def process_data(self):
        for file_path in self.file_paths:
            try:
                mat_data = scipy.io.loadmat(file_path)
                image_data = mat_data["images"]
                # Convert image data to uint8
                image_data = np.uint8(image_data)
                # Convert to PIL Image
                img = Image.fromarray(image_data.squeeze())
                # Convert to PNG and save
                png_path = os.path.splitext(file_path)[0] + ".png"
                img.save(png_path)
                # Display image on canvas
                self.display_image(png_path)
            except Exception as e:
                error_message = f"Error processing data: {e}"
                messagebox.showerror("Error", error_message)

    def display_image(self, image_path):
        try:
            # Open image
            img = Image.open(image_path)
            # Resize image to fit canvas
            img = img.resize((400, 300), Image.ANTIALIAS)
            # Display image on canvas
            self.canvas.img = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas.img)
        except Exception as e:
            error_message = f"Error displaying image: {e}"
            messagebox.showerror("Error", error_message)






    def ElectroMap_OpeningFcn(hObject=None, *args, handles=None, varargin=None):
        # Function for setting values on interface initialization
        handles.output = hObject
        handles.invertopt = 1
        handles.sfilt = 2
        handles.velout = 4
        handles.bgon = 1
        handles.bgcol = 'w'
        handles.folder_name = []
        handles.fname = 'opening of GUI hold'
        handles.lastprocessedfname = 'opening of GUI hold'
        handles.drawcon = 0
        handles.conbon = []
        handles.medifilt = 1
        masterdir = os.getcwd()
        handles.ttpstart = 10
        handles.ttpend = 90
        handles.roinum = 1
        handles.roisum = 0
        handles.snrt1 = 10
        handles.snrt2 = 30
        handles.pbefore = 5
        handles.pafter = 5
        handles.herefromroiload = 0
        handles.rect = []
        handles.loadedmask = []
        handles.herefromsegmentpush = 0
        handles.filming = 0
        
        # Axis not visible until something is in them
        plt.axis('off')
        handles.isZoomed = 0
        handles.fmin = 0.5
        handles.fmax = 10
        handles.fbin = 0.05
        handles.dfwin = 0
    
    # Update handles structure
#     guidata(hObject, handles)
    
    def ElectroMap_OutputFcn(*args, **kwargs):
        # Get default command line output from handles structure
        return handles.output

    def pushselect_Callback(hObject=None, *args, **kwargs):
        handles = guidata(hObject)
        ## Populate listbox with .mat and .tif files
        set(handles.listbox1, 'Value', 1)
        
        handles.folder_name = uigetdir()
        workingdir = handles.folder_name
        os.chdir(workingdir)
        
        # get all files in directory
        allfiles = os.listdir(handles.folder_name)
        addpath(handles.folder_name)
        
        file_list = []
        # Find tif and mat files
        for file in allfiles:
            if file.endswith('.TIF') or file.endswith('.tif') or file.endswith('.mat'):
                file_list.append(file)
        
        set(handles.listbox1, 'String', file_list)
        guidata(hObject, handles)

    import numpy as np
    import os

    def listbox1_Callback(*args, **kwargs):
        pass

    def listbox1_CreateFcn(hObject=None, *args, **kwargs):
        if ispc() and get(hObject, 'BackgroundColor') == get(0, 'defaultUicontrolBackgroundColor'):
            set(hObject, 'BackgroundColor', 'white')

    def pushload_Callback(hObject=None, *args, **kwargs): 
        handles = guidata(hObject)
        set(handles.listbox2, 'Value', 1)
        set(handles.pushprocess, 'Enable', 'on')
        set(handles.producemaps, 'Enable', 'off')
        ## Get image info from GUI
        # image
        handles.threshop = get(handles.threshopt, 'Value')
        handles.threshman = get(handles.manthresh, 'Value')
        imchoice = get(handles.imagedisp, 'Value')
        cropchoice = get(handles.cropbox, 'Value')
        handles.cropchoice = cropchoice
        if cropchoice == 1:
            handles.rect = []
        quinnieopt = get(handles.squareROI, 'Value')
        
        ## file info
        chosenfilecontents = cellstr(get(handles.listbox1, 'String'))
        choice = get(handles.listbox1, 'Value')
        fname = chosenfilecontents[choice]
        if ispc == 1:
            handles.fnamenew = os.path.join(handles.folder_name, '/', fname)
        else:
            handles.fnamenew = os.path.join(handles.folder_name, '/', fname)
        
        tf = str(handles.fname) == str(handles.fnamenew)
        if tf == 0:
            handles.rect = []
            handles.images = []
            handles.lastprocessedfname = 'Pressing load or changing threshold hold'
        handles.fname = handles.fnamenew
        
        ## If custom roi chosen, get rid off manual thresholding
        if quinnieopt == 1 or handles.herefromroiload == 1:
            handles.threshop = 2
            handles.threshman = -50000
            if handles.herefromroiload == 1:
                quinnieopt = 0

        ## load, crop and reset opt, threshold image
    # inversion = get(handles.invertopt, 'Value')
    # camopt = 0
    # axes(handles.imageaxes)
    # set(handles.resegment, 'Enable', 'off')
    # set(handles.B2B, 'Enable', 'off')
    # ## Load new image using OMimload function

    if tf == 0:
        num_images, handles.newrect, mask, im, handles.I, boundaries, handles.camopt, handles.frame1, handles.fluoim, handles.rois, handles.rhsn = OMimload(handles.fname, cropchoice, quinnieopt, handles.threshop, handles.threshman, handles.rect, inversion, camopt, get(handles.imagedisp, 'Value'), handles.roinum, handles.roisum)
        handles.mask = mask
        handles.im = im
        handles.num_images = num_images
        handles.rect = handles.newrect
        if len(handles.loadedmask) == 0 and handles.herefromroiload == 1:
            handles.mask = mask
            handles.I = np.multiply(im, mask)
            handles.boundaries = boundaries
            if im.shape[0] != mask.shape[0] or im.shape[1] != mask.shape[1]:
                if np.abs(im.shape[0] - mask.shape[0]) <= 2 and np.abs(im.shape[1] - mask.shape[1]) <= 2:
                    choice = questdlg('ROI dimensions do not match Image but only slightly off. Would you like to reshape ROI?', 'ROI mismatch', 'Yes', 'No', 'Yes')
                    if choice == 'Yes':
                        rows, cols = im.shape
                        rows2, cols2 = mask.shape
                        if rows2 > rows and cols2 > cols:
                            newmask = np.zeros((rows, cols))
                            newmask[:rows, :cols] = mask[:rows, :cols]
                        if rows > rows2 and cols > cols2:
                            newmask = np.zeros((rows, cols))
                            newmask[:rows2, :cols2] = mask
                        mask = newmask.astype(np.uint16)
                else:
                    handles.herefromroiload = 0
                    handles.loadedmask = []
                    h = errordlg('Loaded ROI dimensions do not match Image')
                    waitfor(h)
        set(handles.cropbox, 'Value', 0)
        handles.boundaries = boundaries

    ## Rethreshold loaded image set
    if tf == 1:
        num_images, handles.newrect, mask, im, handles.I, boundaries, handles.camopt, handles.frame1, handles.fluoim, handles.rois, handles.rhsn = OMimload(handles.fname, cropchoice, quinnieopt, handles.threshop, handles.threshman, handles.rect, inversion, camopt, get(handles.imagedisp, 'Value'), handles.roinum, handles.roisum)
        handles.mask = mask
        handles.im = im
        handles.num_images = num_images
        handles.rect = handles.newrect
        if len(handles.loadedmask) == 0 and handles.herefromroiload == 1:
            handles.mask = mask
            handles.I = np.multiply(im, mask)
            handles.boundaries = boundaries
            if im.shape[0] != mask.shape[0] or im.shape[1] != mask.shape[1]:
                if np.abs(im.shape[0] - mask.shape[0]) <= 2 and np.abs(im.shape[1] - mask.shape[1]) <= 2:
                    choice = questdlg('ROI dimensions do not match Image but only slightly off. Would you like to reshape ROI?', 'ROI mismatch', 'Yes', 'No', 'Yes')
                    if choice == 'Yes':
                        rows, cols = im.shape
                        rows2, cols2 = mask.shape
                        if rows2 > rows and cols2 > cols:
                            newmask = np.zeros((rows, cols))
                            newmask[:rows, :cols] = mask[:rows, :cols]
                        if rows > rows2 and cols > cols2:
                            newmask = np.zeros((rows, cols))
                            newmask[:rows2, :cols2] = mask
                        mask = newmask.astype(np.uint16)
                else:
                    handles.herefromroiload = 0
                    handles.loadedmask = []
                    h = errordlg('Loaded ROI dimensions do not match Image')
                    waitfor(h)
        set(handles.cropbox, 'Value', 0)
        handles.boundaries = boundaries

        ## change pic in GUI
    axes(handles.imageaxes)
    cla()
    if imchoice == 1:
        plt.imshow(handles.frame1, cmap='gray', interpolation='none')
        for i in range(boundaries.shape[0]):
            plt.plot(boundaries[i][:, 1], boundaries[i][:, 0], 'r', linewidth=2)
    elif imchoice == 2:
        plt.imshow(handles.fluoim, cmap='jet', interpolation='none')
        for i in range(boundaries.shape[0]):
            plt.plot(boundaries[i][:, 1], boundaries[i][:, 0], 'k', linewidth=2)

    set(handles.pushprocess, 'Enable', 'on')
    set(handles.producemaps, 'Enable', 'off')
    guidata(hObject, handles)
    # --- Executes on button press in pushprocess.

    def pushprocess_Callback(hObject=None, *args, handles=None):
        minpeakdist = float(get(handles.minpeak, 'String'))
        minpeakdist = np.ceil(minpeakdist / (1 / float(get(handles.framerate, 'String'))))
        handles.minpeakdist = minpeakdist
        segchoice = get(handles.segchoice, 'Value')
        div = float(get(handles.segsize, 'String'))
        minboundary = float(get(handles.minbound, 'String'))
        minmumofpeaks = float(get(handles.minnum, 'String'))
        handles.avgCL = []
        # Baseline
        BLopt = get(handles.BLopt, 'Value')
        # filtering
        tfilt = get(handles.tfilt, 'Value')
        sfilt = get(handles.sfilt, 'Value')
        sfiltsize = float(get(handles.sfiltsize, 'String'))
        # outlieropts
        handles.outlier = get(handles.apdout, 'Value')
        handles.outliervel = get(handles.velout, 'Value')
        # inversion
        inversion = get(handles.invertopt, 'Value')
        # frame removal
        handles.frameremove = get(handles.removef, 'Value')
        ## is same file check and process images
        chosenfilecontents = cellstr(get(handles.listbox1, 'String'))
        choice = get(handles.listbox1, 'Value')
        newfname = chosenfilecontents[choice]
        tf = str(newfname) == str(handles.lastprocessedfname)
        if tf == 0:
            loadnewims = 1
            handles.lastprocessedfname = newfname
        else:
            if tf == 1 and handles.herefromsegmentpush == 0:
                newsettingschoice = questdlg('Re-Process?', 'Re-Load same file', 'Yes', 'No - Just segment', 'No - Just segment')
                if 'Yes' == newsettingschoice:
                    loadnewims = 1
                else:
                    if 'No - Just segment' == newsettingschoice:
                        loadnewims = 0
                        num_images = handles.num_images
            else:
                if tf == 1 and handles.herefromsegmentpush == 1:
                    loadnewims = 0

        if loadnewims == 1:
            if len(handles.rect) == 0:
                handles.cropchoice = 1
            handles.averages = []
            handles.preimages, images, averages, mask = OMimprocess(handles.fname, handles.im, handles.rect, handles.num_images, handles.cropchoice, handles.mask, sfilt, sfiltsize, inversion, tfilt, handles.frameremove, handles.camopt, float(get(handles.sfiltsigma, 'String')), handles.pbefore, handles.pafter, handles.rhsn)
            handles.waverages = averages
            handles.images = images
            handles.averages = averages
            num_images = handles.num_images
            handles.mask = mask

        set(handles.resegment, 'Enable', 'on')
        set(handles.B2B, 'Enable', 'on')
        if loadnewims == 0:
            averages = handles.averages
            images = handles.images
            num_images = handles.num_images

        ## Baseline Drift Correction

        # Top hat filter
        if BLopt in [1, 4]:
            th_len = float(get(handles.thlen, 'String'))
            th_len = th_len / float(get(handles.framerate, 'String'))
            th_len = np.round(th_len)
            se = strel('line', th_len, 0.5)
            BLAV = imopen(averages, se)

        # Poly 4th degree
        if BLopt in [2, 5]:
            p = np.polyfit(np.arange(1, len(averages) + 1), averages, 4)
            BLAV = np.polyval(p, np.arange(1, len(averages) + 1))

        # Poly 11th degree
        if BLopt in [3, 6]:
            p = np.polyfit(np.arange(1, len(averages) + 1), averages, 11)
            BLAV = np.polyval(p, np.arange(1, len(averages) + 1))

        
        # No BL correction
    if BLopt == 7:
        BLAV = np.amin(averages)

    handles.averages = (averages - BLAV)

    ## Remove baseline from each pixel

    BLAV = BLAV - np.amin(BLAV)
    if BLopt in [4, 5, 6]:
        for t in np.arange(1, images.shape[2] + 1):
            images[:, :, t] = images[:, :, t] + BLAV[t]

    wb = waitbar(0.5, 'Removing Baseline')
    signal = np.zeros(images.shape[2])
    if BLopt in [1, 2, 3]:
        for row in np.arange(0, images.shape[0]):
            for col in np.arange(0, images.shape[1]):
                signal = images[row, col, :]
                if inversion == 1:
                    signal = np.invert(signal)
                if BLopt == 1:
                    se = strel('line', th_len, 0.5)
                    BL = imopen(signal, se)
                elif BLopt == 2:
                    p, _, mu = np.polyfit(np.arange(1, len(signal) + 1), signal, 4, full=True)
                    BL = np.polyval(p, np.arange(1, len(signal) + 1), [], mu)
                elif BLopt == 3:
                    p, _, mu = np.polyfit(np.arange(1, len(signal) + 1), signal, 11, full=True)
                    BL = np.polyval(p, np.arange(1, len(signal) + 1), [], mu)
                images[row, col, :] = images[row, col, :] + BL
                images[row, col, :] = images[row, col, :] - np.amin(images[row, col, :])

    handles.images = images
    waitbar(0.95, wb, 'Segmenting Signal')
    wholav = handles.averages
    ## Display Signal
    schoice = 1
    if schoice == 1:
        handles.averages = handles.waverages

    if schoice in [2, 3, 4]:
        if schoice in [2, 4]:

            plt.imshow(handles.frame1, cmap='gray', interpolation='none')
            plt.title('Make your selection and press enter')
            rec = plt.ginput(2)
            cropfig = plt.gcf()
            plt.close(cropfig)
            rec = np.floor(rec).astype(int)
            r1 = rec[0][1]
            c1 = rec[0][0]
            if r1 == 0:
                r1 = 1
            if c1 == 0:
                c1 = 1
            r2 = rec[1][1]
            c2 = rec[1][0]
            ar, ac, numim = handles.images.shape
            rmask = np.zeros((ar, ac))
            for r in np.arange(r1, r2 + 1):
                for c in np.arange(c1, c2 + 1):
                    rmask[r, c] = 1
            rmask = rmask.astype(np.uint16)
            newav = np.zeros(numim)
            for j in np.arange(0, numim):
                roiim = np.multiply(handles.images[:, :, j], rmask)
                newav[j] = np.sum(np.sum(roiim))
            newav = np.invert(newav)
            newav = newav - np.amin(newav)

        if schoice in [3, 4]:
            if schoice == 3:
                newav = handles.waverages
            dnewav = np.convolve(newav, np.ones(5) / 5, mode='same')
            dnewav = np.convolve(dnewav, np.ones(5) / 5, mode='same')
            dnewav[0:10] = 0
            dnewav = dnewav - np.amin(dnewav)
            dnewav[0:10] = 0
            newav = np.array([0, np.transpose(dnewav)])
        # save overall average for later
        wholav = handles.averages
        handles.averages = newav

        
        set(handles.listbox2, 'Value', 1)
    axes(handles.axes2)
    plt.plot(handles.averages)
    drawnow()

    ## BLremoval
    ## Baseline Drift Correction
    if schoice == 1 or schoice == 2:
        if schoice == 1:
            handles.averages = wholav
        if BLopt in [1, 4]:
            th_len = float(get(handles.thlen, 'String'))
            th_len = th_len / float(get(handles.framerate, 'String'))
            th_len = np.round(th_len)
            se = strel('line', th_len, 0.5)
            BLAV = imopen(handles.averages, se)
        elif BLopt in [2, 5]:
            p, _, mu = polyfit(np.arange(1, len(handles.averages) + 1), handles.averages, 4)
            BLAV = polyval(p, np.arange(1, len(handles.averages) + 1), [], mu)
        elif BLopt in [3, 6]:
            p, _, mu = polyfit(np.arange(1, len(handles.averages) + 1), handles.averages, 11)
            BLAV = polyval(p, np.arange(1, len(handles.averages) + 1), [], mu)
        elif BLopt == 7:
            BLAV = np.amin(handles.averages)
        handles.averages = (handles.averages - BLAV)

    axes(handles.axes2)
    plt.plot(handles.averages)
    drawnow()

    ## DETECT PEAKS
    before = float(get(handles.beforeGUI, 'String')) * float(get(handles.framerate, 'String'))
    before = np.round(before)
    after = float(get(handles.afterGUI, 'String')) * float(get(handles.framerate, 'String'))
    after = np.round(after)
    handles.locs = []
    handles.q2locs = []
    handles.avgCL = []
    handles.locs, *argsdles.q2locs, handles.avgCL, handles.numofpeaksoverall, handles.peakheight = Omseg2(handles.averages, float(get(handles.peakhigh, 'String')), minpeakdist, float(get(handles.peakhigh, 'String')), minpeakdist, minboundary, segchoice, minmumofpeaks, num_images, div, before, after)

    ## Zoomed Section
    axes(handles.axes2)
    origInfo = getappdata(gca, 'matlab_graphics_resetplotview')
    handles.isZoomed = 0
    if len(origInfo) == 0:
        handles.isZoomed = 0
    else:
        handles.isZoomed = 1

    exposure = 1 / float(get(handles.framerate, 'String'))
    handles.newlim = get(gca, 'XLim') / exposure
    handles.newlim[0] = int(np.floor(handles.newlim[0]))
    handles.newlim[1] = np.ceil(handles.newlim[1])

    ## axes2
    handles.avgCL = np.multiply(handles.avgCL, (1 / float(get(handles.framerate, 'String'))))
    axes(handles.axes2)
    cla()
    CM = np.array(['b', 'r', 'g', 'y', 'c', 'm', 'k'])
    exposure = 1 / float(get(handles.framerate, 'String'))
    handles.averagestime = (np.arange(0, len(handles.averages), 1)) * exposure
    plt.plot(handles.averagestime, handles.averages, 'k')
    plt.xlabel('time (ms) \rightarrow')
    plt.ylabel('Fluorescence Intensity')
    plt.xlim(np.array([0, len(handles.averages) * exposure]))
    hold('on')
    plt.plot(handles.averagestime[handles.locs], handles.averages[handles.locs], 'or')
    before = np.round(float(get(handles.beforeGUI, 'String')) / exposure)
    after = np.round(float(get(handles.afterGUI, 'String')) / exposure)
    if len(handles.locs) < 2:
        handles.q2locs = handles.locs
        handles.avgCL[1, 0] = 0

    for i in range(len(handles.q2locs[0])):
        c = np.mod(i, 6)
        if c == 0:
            c = 6
        A = handles.q2locs[0][i]
        if len(A) == 1:
            errordlg('No constant cycle length regions found. Please adjust pre-process settings')
        if np.amin(A[A > 0]) < before:
            k = np.where(A > 0)
            A[k[0][1]] = 0
        tstart = np.amin(A[A > 0]) - before
        if tstart == 0:
            tstart = 1
        tend = np.amax(A) + after
        if tend > len(handles.averagestime):
            if len(A) > 1:
                tend = A[-2] + after
            else:
                if len(A) == 1:
                    tend = len(handles.averagestime)
                    c = 7
        if tend > len(handles.averagestime):
            tend = len(handles.averagestime)
        plt.plot(handles.averagestime[int(tstart):int(tend) + 1], handles.averages[int(tstart):int(tend) + 1], color=CM[c])

    # end
    colorbar('off')
    hold('off')
    set(gca, 'FontSize', 8)
    ax = gca()
    line(get(ax, 'XLim'), np.array([handles.peakheight, handles.peakheight]), 'Color', 'b')
    # populate section listbox

    section = []
    if handles.numofpeaksoverall == 1:
        section.append('N/A')
    else:
        for i in range(len(handles.q2locs[0])):
            section.append(num2str(i) + ' (' + num2str((handles.avgCL[1][i])) + 'ms)')

    ## add zoomed section
    if handles.isZoomed == 1:
        if len(handles.q2locs[0]) < 2:
            handles.q2locs[:, -1] = 0
        newline = np.zeros((1, len(handles.q2locs[0])))
        newline[0] = handles.newlim[0]
        newline[1] = handles.newlim[1]
        handles.q2locs = np.vstack((handles.q2locs, newline))
        newsection = 'Zoomed Section'
        section.append(newsection)

    handles.section = section
    ## Couple sections to auto windows
    # peak count set to 1 because of first ignored peak, should be changes
    handles.winopt = 1
    if len(handles.locs) == 1 or len(handles.locs) == 2:
        if len(handles.locs) == 1:
            handles.q2locs[0][0] = handles.locs
        if len(handles.locs) == 2:
            handles.q2locs = handles.locs
            CL2 = handles.locs[1] - handles.locs[0] * exposure
            handles.section[0] = num2str(CL2) + 'ms'
            section = handles.section

    set(handles.listbox2, 'String', section)
    axes(handles.mapaxes)
    pretty = get(handles.colmap, 'String')
    jetcolormap = colormap(pretty[get(handles.colmap, 'Value')])
    jetcolormap[0] = np.array([1, 1, 1])
    colormap(jetcolormap)
    os.delete(wb)
    if schoice == 2 or schoice == 3:
        handles.averages = wholav

    set(handles.pushprocess, 'Enable', 'on')
    set(handles.producemaps, 'Enable', 'on')
    guidata(hObject, handles)
    # --- Executes on selection change in listbox2.

        
    def listbox2_Callback(hObject=None, eventdata=None, *args):
        handles = guidata(hObject)
        axes(handles.axes2)
        # handles.filming = 0;
        exposure = 1 / float(get(handles.framerate, 'String'))
        handles.averagestime = np.arange(0, len(handles.averages) + 1) * exposure
        plt.plot(handles.averagestime, handles.averages, 'k')
        plt.xlabel('time (ms) \rightarrow')
        plt.ylabel('Fluorescence Intensity')
        plt.xlim([0, len(handles.averages) * exposure])
        plt.hold(True)
        plt.plot(handles.averagestime[handles.locs], handles.averages[handles.locs], 'or')
        before = np.round(float(get(handles.beforeGUI, 'String')) / exposure)
        after = np.round(float(get(handles.afterGUI, 'String')) / exposure)
        section_choice = get(handles.listbox2, 'Value')
        A = handles.q2locs[section_choice, :]
        if len(A) == 0:
            errordlg('No constant cycle length regions found. Please adjust pre-process settings')
        
        if np.amin(A(A > 0)) < before:
            k = find(A)
            A[k[1]] = 0
        
        tstart = np.amin(A(A > 0)) - before
        if tstart == 0:
            tstart = 1
        
        tend = np.amax(A) + after
        if tend > len(handles.averagestime):
            tend = len(handles.averagestime)
        
        plt.plot(handles.averagestime(np.arange(tstart,tend+1)),handles.averages(np.arange(tstart,tend+1)),'color','r')
        ax = gca
        line(get(ax,'XLim'),np.array([handles.peakheight,handles.peakheight]),'Color','b')
        set(gca,'FontSize',8)
        producemaps_Callback(hObject,eventdata,handles)
        # --- Executes during object creation, after setting all properties.
        
    def listbox2_CreateFcn(hObject = None,*argsne,**kwargs): 
        # hObject    handle to listbox2 (see GCBO)
    # eventdata  reserved - to be defined in a future version of MATLAB
    # handles    empty - handles not created until after all CreateFcns called
        
        # Hint: listbox controls usually have a white background on Windows.
    #       See ISPC and COMPUTER.
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on selection change in tfilt.
        
    def segchoice_Callback(*argsne,**kwargs): 
        pass
        # --- Executes during object creation, after setting all properties.
        
    def segchoice_CreateFcn(hObject=None):
        if ispc and get(hObject, 'BackgroundColor') == get(0, 'defaultUicontrolBackgroundColor'):
            set(hObject, 'BackgroundColor', 'white')

        
    # def segsize_Callback(*argsne,*argsne,*argsne): 
    #     pass
        # --- Executes during object creation, after setting all properties.
        
    def segsize_CreateFcn(hObject=None, *args):
        if ispc and get(hObject, 'BackgroundColor') == get(0, 'defaultUicontrolBackgroundColor'):
            set(hObject, 'BackgroundColor', 'white')
            
        
        # --- Executes on selection change in BLopt.
        
    # def BLopt_Callback(*argsne,*argsne,*argsne): 
    #     pass
        # --- Executes during object creation, after setting all properties.
        
    def BLopt_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on selection change in tfilt.
        
    # def tfilt_Callback(*argsne,*argsne,*argsne): 
    #     pass
    #     # --- Executes during object creation, after setting all properties.
        
    def tfilt_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    def minpeak_Callback(*argsne,**kwargs): 
        pass
        # --- Executes during object creation, after setting all properties.
        
    def minpeak_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    def minnum_Callback(*argsne,**kwargs): 
        pass
        # --- Executes during object creation, after setting all properties.
        
    def minnum_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    # def minbound_Callback(*argsne,*argsne,*argsne): 
    #     pass
    #     # --- Executes during object creation, after setting all properties.
        
    def minbound_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on button press in invertopt.
        
    # def invertopt_Callback(*argsne,*argsne,*argsne): 
    #     pass
        # hObject    handle to invertopt (see GCBO)
    # eventdata  reserved - to be defined in a future version of MATLAB
    # handles    structure with handles and user data (see GUIDATA)
        
        # Hint: get(hObject,'Value') returns toggle state of invertopt
        
        # --- Executes on selection change in sfilt.
        
    # def sfilt_Callback(*argsne,*argsne,*argsne):
    #     pass
        # --- Executes during object creation, after setting all properties.
        
    def sfilt_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    # def sfiltsize_Callback(*argsne,**kwargs): 
    #     pass
        # --- Executes during object creation, after setting all properties.
        
    def sfiltsize_CreateFcn(hObject = None,*argsne,**kwargs): 
        # hObject    handle to sfiltsize (see GCBO)
    # eventdata  reserved - to be defined in a future version of MATLAB
    # handles    empty - handles not created until after all CreateFcns called
        
        # Hint: edit controls usually have a white background on Windows.
    #       See ISPC and COMPUTER.
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on selection change in Mapchoice.
        
    def Mapchoice_Callback(hObject = None,*argsne,**kwargs): 
        # hObject    handle to Mapchoice (see GCBO)
    # eventdata  reserved - to be defined in a future version of MATLAB
    # handles    structure with handles and user data (see GUIDATA)
        handles = guidata(hObject)
        choice = get(handles.Mapchoice,'Value')
        axes(handles.bgimage)
        if handles.bgon == 0:
            imshow(handles.frame1,[],'InitialMagnification',400)
            colormap('gray')
            freezeColors
        else:
            cla('reset')
            plt.axis('off')
        
        rows,cols = handles.frame1.shape
        axes(handles.mapaxes)
        cla('reset')
        plt.axis('off')
        drawnow()
        imshow(np.zeros((rows,cols)))
        isochoice = get(handles.isoopt,'Value')
        CVmap = []
        if choice == 10:
            cla('reset')
            set(handles.meanchange,'String','')
            set(handles.textchange,'String','')
            # map,*args,*argsrs(handles.averageBeat,handles.mask,handles.snrt1,handles.snrt2,(get(handles.tfilt,'Value')))
            if len(map)==0 == 1:
                map = 0
                alll = 0
            dmap = map
            pretty = get(handles.colmap,'String')
            jetcolormap = (colormap(pretty[get(handles.colmap,'Value')]))
            colormap(jetcolormap)
            if get(handles.apdscale,'Value') == 1:
                dmap[dmap == 0] = NaN
                him = imshow(dmap,[],'InitialMagnification',800)
                set(him,'AlphaData',not np.isnan(dmap) )
                if handles.bgon == 1:
                    plt.axis('on')
                    set(gca,'XColor','none','yColor','none','xtick',[],'ytick',[],'Color',handles.bgcol)
                caxis(np.array([int(np.floor(np.amin(np.amin(alll)))),np.ceil(np.amax(np.amax(alll)))]))
            if get(handles.apdscale,'Value') == 2:
                dmap[dmap == 0] = NaN
                him = imshow(dmap,np.array([str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String'))]),'InitialMagnification',800)
                set(him,'AlphaData',not np.isnan(dmap) )
                if handles.bgon == 1:
                    plt.axis('on')
                    set(gca,'XColor','none','yColor','none','xtick',[],'ytick',[],'Color',handles.bgcol)
                caxis(np.array([str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String'))]))
            plt.title('SNR Map')
            freezeColors
            # colorbar
            axes(handles.cb)
            cla('reset')
            hcb = colorbar
            pretty = get(handles.colmap,'String')
            colormap(colormap(pretty[get(handles.colmap,'Value')]))
            hcb.Location = 'southoutside'
            cpos = hcb.Position
            cpos[4] = 4 * cpos(4)
            hcb.Position = cpos
            hcb.TicksMode = 'manual'
            hcb.TickLabelsMode = 'manual'
            if get(handles.apdscale,'Value') == 1:
                stepp = (np.ceil(np.amax(np.amax(alll))) - int(np.floor(np.amin(np.amin(alll))))) / 5
                hcb.TickLabels = (np.arange(int(np.floor(np.amin(np.amin(alll)))),np.ceil(np.amax(np.amax(alll)))+stepp,stepp))
                hcb.Ticks = np.array([0.01,np.arange(0.2,1+0.2,0.2)])
                hcb.Label.String = 'Signal/Noise'
                plt.axis('off')
            if get(handles.apdscale,'Value') == 2:
                stepp = (str2double(get(handles.cmax,'String')) - str2double(get(handles.cmin,'String'))) / 5
                hcb.TickLabels = np.array([np.arange(str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String'))+stepp,stepp)])
                hcb.Ticks = np.array([0.01,np.arange(0.2,1+0.2,0.2)])
                hcb.Label.String = 'Signal/Noise'
                plt.axis('off')
            pos = get(hcb,'Position')
            hcb.Label.Position = np.array([pos(1),pos(2) - 1.2])
            stdall = std(alll)
            palll = prctile(alll,np.array([5,50,95]))
            handles.rdata[4,1] = mean(alll)
            handles.rdata[4,2] = stdall
            handles.rdata[4,3] = stdall / np.sqrt(np.asarray(alll).size)
            handles.rdata[4,4] = stdall * stdall
            handles.rdata[4,5] = ((palll(3) - palll(1)) / palll(2))
            rownames[0] = 'APD'
            rownames[2] = 'CV'
            rownames[3] = 'Amp'
            rownames[4] = 'SNR'
            axes(handles.mapaxes)
            set(handles.rtable,'RowName',rownames)
            set(handles.rtable,'data',handles.rdata)
        
        if choice == 1:
            cla('reset')
            set(handles.meanchange,'String','')
            set(handles.textchange,'String','')
            t = str2double(get(handles.t,'String'))
            map = handles.apdmap
            alll = handles.apalll
            if len(map)==0 == 1:
                map = 0
                alll = 0
            dmap = map
            pretty = get(handles.colmap,'String')
            jetcolormap = (colormap(pretty[get(handles.colmap,'Value')]))
            colormap(jetcolormap)
            if get(handles.apdscale,'Value') == 1:
                dmap[dmap == 0] = NaN
                him = imshow(dmap,[],'InitialMagnification',800)
                set(him,'AlphaData',not np.isnan(dmap) )
                if handles.bgon == 1:
                    plt.axis('on')
                    set(gca,'XColor','none','yColor','none','xtick',[],'ytick',[],'Color',handles.bgcol)
                caxis(np.array([int(np.floor(np.amin(np.amin(alll)))),np.ceil(np.amax(np.amax(alll)))]))
            if get(handles.apdscale,'Value') == 2:
                dmap[dmap == 0] = NaN
                him = imshow(dmap,np.array([str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String'))]),'InitialMagnification',800)
                set(him,'AlphaData',not np.isnan(dmap) )
                if handles.bgon == 1:
                    plt.axis('on')
                    set(gca,'XColor','none','yColor','none','xtick',[],'ytick',[],'Color',handles.bgcol)
                caxis(np.array([str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String'))]))
            freezeColors
            # colorbar
            axes(handles.cb)
            cla('reset')
            hcb = colorbar
            pretty = get(handles.colmap,'String')
            colormap(colormap(pretty[get(handles.colmap,'Value')]))
            hcb.Location = 'southoutside'
            cpos = hcb.Position
            cpos[4] = 4 * cpos(4)
            hcb.Position = cpos
            hcb.TicksMode = 'manual'
            hcb.TickLabelsMode = 'manual'
            if get(handles.apdscale,'Value') == 1:
                stepp = (np.ceil(np.amax(np.amax(alll))) - int(np.floor(np.amin(np.amin(alll))))) / 5
                hcb.TickLabels = np.array([np.arange(int(np.floor(np.amin(np.amin(alll)))),np.ceil(np.amax(np.amax(alll)))+stepp,stepp)])
                hcb.Ticks = np.array([0.01,np.arange(0.2,1+0.2,0.2)])
                hcb.Label.String = 'Duration (ms)'
                plt.axis('off')
            if get(handles.apdscale,'Value') == 2:
                stepp = (str2double(get(handles.cmax,'String')) - str2double(get(handles.cmin,'String'))) / 5
                hcb.TickLabels = np.array([np.arange(str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String'))+stepp,stepp)])
                hcb.Ticks = np.array([0.01,np.arange(0.2,1+0.2,0.2)])
                hcb.Label.String = 'Duration (ms)'
                plt.axis('off')
            pos = get(hcb,'Position')
            hcb.Label.Position = np.array([pos(1),pos(2) - 1.2])
            stdall = std(alll)
            palll = prctile(alll,np.array([5,50,95]))
            handles.rdata[1,1] = mean(alll)
            handles.rdata[1,2] = stdall
            handles.rdata[1,3] = stdall / np.sqrt(np.asarray(alll).size)
            handles.rdata[1,4] = stdall * stdall
            handles.rdata[1,5] = ((palll(3) - palll(1)) / palll(2))
            *argsNRr,allSNRdb = snrs(handles.averageBeat,handles.mask,handles.snrt1,handles.snrt2,(get(handles.tfilt,'Value')))
            handles.rdata[4,1] = mean(allSNRr)
            handles.rdata[4,2] = mean(allSNRdb)
            rownames = get(handles.rtable,'RowName')
            rownames[4] = 'SNR'
            axes(handles.mapaxes)
            plt.title(np.array(['APD',num2str(t)]))
            set(handles.rtable,'RowName',rownames)
            set(handles.rtable,'data',handles.rdata)
        
        if choice == 2:
            cla
            set(handles.meanchange,'String','')
            set(handles.textchange,'String','')
            map = handles.actmap
            if isochoice == 1:
                mini = 0
                maxi = np.amax(np.amax(map))
            else:
                if isochoice == 2:
                    mini = str2double(get(handles.isomin,'String'))
                    maxi = str2double(get(handles.isomax,'String'))
            dmap = map
            dmap[dmap == 0] = NaN
            him = imshow(dmap,np.array([0,maxi]),'InitialMagnification',800)
            hold('on')
            pretty = get(handles.colmap,'String')
            jetcolormap = (colormap(pretty[get(handles.colmap,'Value')]))
            colormap(jetcolormap)
            caxis(np.array([mini,maxi]))
            set(him,'AlphaData',not np.isnan(dmap) )
            if handles.bgon == 1:
                plt.axis('on')
                set(gca,'XColor','none','yColor','none','xtick',[],'ytick',[],'Color',handles.bgcol)
            plt.title('Activation Map')
            freezeColors
            #colorbar
            axes(handles.cb)
            cla('reset')
            hcb = colorbar
            hcb.Location = 'southoutside'
            cpos = hcb.Position
            cpos[4] = 4 * cpos(4)
            hcb.Position = cpos
            hcb.TicksMode = 'manual'
            hcb.TickLabelsMode = 'manual'
            stepp = (maxi - mini) / 5
            hcb.TickLabels = np.array([np.arange(mini,maxi+stepp,stepp)])
            hcb.Ticks = np.array([0.01,np.arange(0.2,1+0.2,0.2)])
            hcb.Label.String = 'Time of activation (ms)'
            plt.axis('off')
            handles.rdata[4,1] = NaN
            handles.rdata[4,2] = NaN
            handles.rdata[4,3] = NaN
            handles.rdata[4,4] = NaN
            handles.rdata[4,4] = NaN
            handles.rdata[4,5] = NaN
            rownames = get(handles.rtable,'RowName')
            rownames[4] = ''
            set(handles.rtable,'RowName',rownames)
            set(handles.meanchange,'String',rownames[4])
            set(handles.rtable,'data',handles.rdata)
        
        if choice == 3:
            cla
            set(handles.meanchange,'String','')
            set(handles.textchange,'String','')
            map = handles.actmap
            quivers_X = handles.quivers_X
            quivers_Y = handles.quivers_Y
            quivers_vx = handles.quivers_vx
            quivers_vy = handles.quivers_vy
            if isochoice == 1:
                mini = 0
                maxi = np.amax(np.amax(map))
            else:
                if isochoice == 2:
                    mini = str2double(get(handles.isomin,'String'))
                    maxi = str2double(get(handles.isomax,'String'))
            dmap = map
            dmap[dmap == 0] = NaN
            him = imshow(dmap,np.array([0,maxi]),'InitialMagnification',800)
            hold('on')
            pretty = get(handles.colmap,'String')
            jetcolormap = (colormap(pretty[get(handles.colmap,'Value')]))
            colormap(jetcolormap)
            caxis(np.array([mini,maxi]))
            set(him,'AlphaData',not np.isnan(dmap) )
            if handles.bgon == 1:
                plt.axis('on')
                set(gca,'XColor','none','yColor','none','xtick',[],'ytick',[],'Color',handles.bgcol)
            scal = str2double(get(handles.scal,'String'))
            hold('on')
            plt.quiver(quivers_X,quivers_Y,scal * quivers_vx,scal * quivers_vy,0,'k')
            # Compile CV map
    # factor=str2double(get(handles.pixelsize,'String'))/10;
    # [rows,cols] = size(map);
    # CVXmap=zeros(rows,cols);
    # CVXmap(sub2ind(size(CVXmap),quivers_Y,quivers_X)) = quivers_vx;
    # sCVXmap=CVXmap.*CVXmap;
    # CVYmap=zeros(rows,cols);
    # CVYmap(sub2ind(size(CVYmap),quivers_Y,quivers_X)) = quivers_vy;
    # sCVYmap=CVYmap.*CVYmap;
    # #construct cv map
    # CVmap=sqrt(sCVXmap+sCVYmap);
    # CVmap=CVmap*factor;
            hold('off')
            plt.title('Activation Map')
            freezeColors
            #colorbar
            axes(handles.cb)
            cla('reset')
            hcb = colorbar
            hcb.Location = 'southoutside'
            cpos = hcb.Position
            cpos[4] = 4 * cpos(4)
            hcb.Position = cpos
            hcb.TicksMode = 'manual'
            hcb.TickLabelsMode = 'manual'
            stepp = (maxi - mini) / 5
            hcb.TickLabels = np.array([np.arange(mini,maxi+stepp,stepp)])
            hcb.Ticks = np.array([0.01,np.arange(0.2,1+0.2,0.2)])
            hcb.Label.String = 'Time of activation (ms)'
            plt.axis('off')
            handles.rdata[4,1] = NaN
            handles.rdata[4,2] = NaN
            handles.rdata[4,3] = NaN
            handles.rdata[4,4] = NaN
            handles.rdata[4,4] = NaN
            handles.rdata[4,5] = NaN
            rownames = get(handles.rtable,'RowName')
            rownames[4] = ''
            set(handles.rtable,'RowName',rownames)
            set(handles.meanchange,'String',rownames[4])
            set(handles.rtable,'data',handles.rdata)
            map = CVmap
        
        if choice == 4:
            cla
            set(handles.meanchange,'String','')
            set(handles.textchange,'String','')
            map = handles.actmap
            quivers_Xout = handles.quivers_Xout
            quivers_Yout = handles.quivers_Yout
            quivers_vxout = handles.quivers_vxout
            quivers_vyout = handles.quivers_vyout
            if isochoice == 1:
                mini = 0
                maxi = np.amax(np.amax(map))
            else:
                if isochoice == 2:
                    mini = str2double(get(handles.isomin,'String'))
                    maxi = str2double(get(handles.isomax,'String'))
            axes(handles.mapaxes)
            dmap = map
            dmap[dmap == 0] = NaN
            him = imshow(dmap,np.array([0,maxi]),'InitialMagnification',800)
            hold('on')
            pretty = get(handles.colmap,'String')
            jetcolormap = (colormap(pretty[get(handles.colmap,'Value')]))
            colormap(jetcolormap)
            caxis(np.array([mini,maxi]))
            set(him,'AlphaData',not np.isnan(dmap) )
            if handles.bgon == 1:
                plt.axis('on')
                set(gca,'XColor','none','yColor','none','xtick',[],'ytick',[],'Color',handles.bgcol)
            scal = str2double(get(handles.scal,'String'))
            hold('on')
            plt.quiver(quivers_Xout,quivers_Yout,scal * quivers_vxout,scal * quivers_vyout,0,'k')
            hold('off')
            # Compile CV map
    # factor=str2double(get(handles.pixelsize,'String'))/10;
    # [rows cols] = size(map);
    # CVmap=zeros(rows,cols);
    # CVXmap=zeros(rows,cols);
    # CVXmap(sub2ind(size(CVXmap),quivers_Yout,quivers_Xout)) = quivers_vxout;
    # sCVXmap=CVXmap.*CVXmap;
    # CVYmap=zeros(rows,cols);
    # CVYmap(sub2ind(size(CVYmap),quivers_Yout,quivers_Xout)) = quivers_vyout;
    # sCVYmap=CVYmap.*CVYmap;
    # #construct cv map
    # CVmap=sqrt(sCVXmap+sCVYmap);
    # CVmap=CVmap*factor;
            plt.title('Activation Map')
            freezeColors
            axes(handles.cb)
            cla('reset')
            hcb = colorbar
            hcb.Location = 'southoutside'
            cpos = hcb.Position
            cpos[4] = 4 * cpos(4)
            hcb.Position = cpos
            hcb.TicksMode = 'manual'
            hcb.TickLabelsMode = 'manual'
            stepp = (maxi - mini) / 5
            hcb.TickLabels = np.array([np.arange(mini,maxi+stepp,stepp)])
            hcb.Ticks = np.array([0.01,np.arange(0.2,1+0.2,0.2)])
            hcb.Label.String = 'Time of activation (ms)'
            plt.axis('off')
            handles.rdata[4,1] = NaN
            handles.rdata[4,2] = NaN
            handles.rdata[4,3] = NaN
            handles.rdata[4,4] = NaN
            handles.rdata[4,4] = NaN
            handles.rdata[4,5] = NaN
            rownames = get(handles.rtable,'RowName')
            rownames[4] = ''
            set(handles.rtable,'RowName',rownames)
            set(handles.meanchange,'String',rownames[4])
            set(handles.rtable,'data',handles.rdata)
            map = CVmap
        
        if choice == 5:
            cla
            axes(handles.mapaxes)
            plt.axis('off')
            #wb=waitbar(0.9,'Calculating Frequencies');
            tic
            map = domfreq(handles.mask,handles.imagerange,str2double(get(handles.framerate,'String')),handles.fmin,handles.fmax,handles.fbin,handles.dfwin,get(handles.tfilt,'Value'))
            toc
            dfs = map(map > 0)
            dmap = map
            if get(handles.apdscale,'Value') == 1:
                dmap[dmap == 0] = NaN
                him = imshow(dmap,np.array([np.amin(np.amin(dfs)),np.amax(np.amax(dfs))]),'InitialMagnification',800)
                set(him,'AlphaData',not np.isnan(dmap) )
                if handles.bgon == 1:
                    plt.axis('on')
                    set(gca,'XColor','none','yColor','none','xtick',[],'ytick',[],'Color',handles.bgcol)
                caxis(np.array([int(np.floor(np.amin(np.amin(dfs)))),np.ceil(np.amax(np.amax(dfs)))]))
                plt.title('Dominant Frequency Map')
                axes(handles.cb)
                cla('reset')
                hcb = colorbar
                hcb.Location = 'southoutside'
                cpos = hcb.Position
                cpos[4] = 4 * cpos(4)
                hcb.Position = cpos
                hcb.TicksMode = 'manual'
                hcb.TickLabelsMode = 'manual'
                hcb.TickLabels = np.array([np.amin(np.amin(dfs)),np.amax(np.amax(dfs))])
                hcb.Ticks = np.array([0,1])
                hcb.Label.String = 'Dominant Frequency (Hz)'
            pretty = get(handles.colmap,'String')
            jetcolormap = (colormap(pretty[get(handles.colmap,'Value')]))
            colormap(jetcolormap)
            if get(handles.apdscale,'Value') == 2:
                dmap[dmap == 0] = NaN
                him = imshow(dmap,np.array([str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String'))]),'InitialMagnification',800)
                set(him,'AlphaData',not np.isnan(dmap) )
                if handles.bgon == 1:
                    plt.axis('on')
                    set(gca,'XColor','none','yColor','none','xtick',[],'ytick',[],'Color',handles.bgcol)
                caxis(np.array([str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String'))]))
                axes(handles.cb)
                cla('reset')
                hcb = colorbar
                hcb.Location = 'southoutside'
                cpos = hcb.Position
                cpos[4] = 4 * cpos(4)
                hcb.Position = cpos
                hcb.TicksMode = 'manual'
                hcb.TickLabelsMode = 'manual'
                hcb.TickLabels = np.array([str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String'))])
                hcb.Ticks = np.array([0,1])
                hcb.Label.String = 'Dominant Frequency (Hz)'
            freezeColors
            plt.axis('off')
            dfs = map(map > 0)
            dfsp = prctile(dfs,np.array([5,50,95]))
            handles.rdata[4,1] = mean(dfs)
            handles.rdata[4,2] = std(dfs)
            handles.rdata[4,3] = std(dfs) / np.asarray(dfs).size
            handles.rdata[4,4] = std(dfs) * std(dfs)
            handles.rdata[4,4] = std(dfs) * std(dfs)
            handles.rdata[4,5] = ((dfsp(3) - dfsp(1)) / dfsp(2))
            rownames = get(handles.rtable,'RowName')
            rownames[4] = 'DF'
            set(handles.rtable,'RowName',rownames)
            set(handles.rtable,'data',handles.rdata)
        
        if choice == 6:
            wb = waitbar(0.5, 'Producing Diastolic Map')
            section_choice = get(handles.listbox2, 'Value')
            A = handles.q2locs[section_choice, :]
            A = A[A != 0]
            frame_1 = A[0]
            frame_last = A[-1]
            exposure = 1 / str2double(get(handles.framerate,'String'))
            after = str2double(get(handles.afterGUI,'String'))
            after = np.round(after / exposure)
            if frame_last + after > handles.num_images:
                frame_last = A(end() - 1)
            map,*args = DInt(str2double(get(handles.framerate,'String')),handles.I,handles.images,handles.outlier,str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String')),get(handles.tfilt,'Value'),str2double(get(handles.beforeGUI,'String')),str2double(get(handles.afterGUI,'String')),str2double(get(handles.minpeak,'String')),frame_1,frame_last,str2double(get(handles.t,'String')))
            if len(alll)==0 == 1:
                map = 0
                alll = 0
            np.asarray(alll).size
            map[np.isnan[map]] = 0
            axes(handles.mapaxes)
            imshow(map,[],'InitialMagnification',800)
            plt.title(np.array(['Diastolic Interval Distribution']))
            pretty = get(handles.colmap,'String')
            jetcolormap = (colormap(pretty[get(handles.colmap,'Value')]))
            jetcolormap[1,:] = np.array([1,1,1])
            colormap(jetcolormap)
            if get(handles.apdscale,'Value') == 1:
                caxis(np.array([int(np.floor(np.amin(np.amin(alll)))),np.ceil(np.amax(np.amax(alll)))]))
            if get(handles.apdscale,'Value') == 2:
                caxis(np.array([str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String'))]))
            freezeColors
            # colorbar
            axes(handles.cb)
            cla('reset')
            hcb = colorbar
            hcb.Location = 'southoutside'
            cpos = hcb.Position
            cpos[4] = 4 * cpos(4)
            hcb.Position = cpos
            hcb.TicksMode = 'manual'
            hcb.TickLabelsMode = 'manual'
            if get(handles.apdscale,'Value') == 1:
                stepp = (np.ceil(np.amax(np.amax(alll))) - int(np.floor(np.amin(np.amin(alll))))) / 5
                hcb.TickLabels = (np.arange(int(np.floor(np.amin(np.amin(alll)))),np.ceil(np.amax(np.amax(alll)))+stepp,stepp))
                hcb.Ticks = np.array([0.01,np.arange(0.2,1+0.2,0.2)])
                hcb.Label.String = 'Diastolic Interval (ms)'
                plt.axis('off')
            if get(handles.apdscale,'Value') == 2:
                stepp = (str2double(get(handles.cmax,'String')) - str2double(get(handles.cmin,'String'))) / 5
                hcb.TickLabels = np.array([np.arange(str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String'))+stepp,stepp)])
                hcb.Ticks = np.array([0.01,np.arange(0.2,1+0.2,0.2)])
                hcb.Label.String = 'Diastolic Interval (ms)'
                plt.axis('off')
            dis = alll
            disper = prctile(dis,np.array([5,50,95]))
            handles.rdata[4,1] = mean(dis)
            handles.rdata[4,2] = std(dis)
            handles.rdata[4,3] = std(dis) / np.asarray(dis).size
            handles.rdata[4,4] = std(dis) * std(dis)
            handles.rdata[4,4] = std(dis) * std(dis)
            handles.rdata[4,5] = ((disper(3) - disper(1)) / disper(2))
            rownames = get(handles.rtable,'RowName')
            rownames[4] = 'DI'
            set(handles.rtable,'RowName',rownames)
            set(handles.rtable,'data',handles.rdata)
            os.delete(wb)
        
        if choice == 7:
            cla
            t = str2double(get(handles.t,'String'))
            map,*args = ttpnew(handles.ttpstart,handles.ttpend,str2double(get(handles.framerate,'String')),t,handles.I,handles.images,handles.averageBeat,handles.outlier,str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String')),get(handles.tfilt,'Value'),str2double(get(handles.beforeGUI,'String')),get(handles.apdbl,'Value'),str2double(get(handles.apdblnum,'String')))
            dmap = map
            plt.title('TTP')
            pretty = get(handles.colmap,'String')
            jetcolormap = (colormap(pretty[get(handles.colmap,'Value')]))
            colormap(jetcolormap)
            dmap[dmap == 0] = NaN
            if get(handles.apdscale,'Value') == 1:
                dmap[dmap == 0] = NaN
                him = imshow(dmap,[],'InitialMagnification',800)
                set(him,'AlphaData',not np.isnan(dmap) )
                if handles.bgon == 1:
                    plt.axis('on')
                    set(gca,'XColor','none','yColor','none','xtick',[],'ytick',[],'Color',handles.bgcol)
                caxis(np.array([int(np.floor(np.amin(np.amin(alll)))),np.ceil(np.amax(np.amax(alll)))]))
            if get(handles.apdscale,'Value') == 2:
                dmap[dmap == 0] = NaN
                him = imshow(dmap,np.array([str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String'))]),'InitialMagnification',800)
                set(him,'AlphaData',not np.isnan(dmap) )
                if handles.bgon == 1:
                    plt.axis('on')
                    set(gca,'XColor','none','yColor','none','xtick',[],'ytick',[],'Color',handles.bgcol)
                caxis(np.array([str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String'))]))
            plt.title('Time to peak Map')
            freezeColors
            # colorbar
            axes(handles.cb)
            cla('reset')
            hcb = colorbar
            hcb.Location = 'southoutside'
            cpos = hcb.Position
            cpos[4] = 4 * cpos(4)
            hcb.Position = cpos
            hcb.TicksMode = 'manual'
            hcb.TickLabelsMode = 'manual'
            if get(handles.apdscale,'Value') == 1:
                np.amax(np.amax(alll))
                np.amin(np.amin(alll))
                stepp = (np.ceil(np.amax(np.amax(alll))) - int(np.floor(np.amin(np.amin(alll))))) / 5
                hcb.TickLabels = np.array([np.arange(int(np.floor(np.amin(np.amin(alll)))),np.ceil(np.amax(np.amax(alll)))+stepp,stepp)])
                hcb.Ticks = np.array([0.01,np.arange(0.2,1+0.2,0.2)])
                hcb.Label.String = 'Duration (ms)'
                plt.axis('off')
            if get(handles.apdscale,'Value') == 2:
                stepp = (str2double(get(handles.cmax,'String')) - str2double(get(handles.cmin,'String'))) / 5
                hcb.TickLabels = np.array([np.arange(str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String'))+stepp,stepp)])
                hcb.Ticks = np.array([0.01,np.arange(0.2,1+0.2,0.2)])
                hcb.Label.String = 'Time to Peak (ms)'
                plt.axis('off')
            ttps = alll
            ttpsp = prctile(ttps,np.array([5,50,95]))
            handles.rdata[4,1] = mean(ttps)
            handles.rdata[4,2] = std(ttps)
            handles.rdata[4,3] = std(ttps) / np.asarray(ttps).size
            handles.rdata[4,4] = std(ttps) * std(ttps)
            handles.rdata[4,4] = std(ttps) * std(ttps)
            handles.rdata[4,5] = ((ttpsp(3) - ttpsp(1)) / ttpsp(2))
            rownames = get(handles.rtable,'RowName')
            rownames[4] = 'TTP'
            set(handles.rtable,'RowName',rownames)
            set(handles.rtable,'data',handles.rdata)
        
        if choice == 8:
            cla
            wb = waitbar(0.5,'Producing tau map')
            # map,*args,*argsutest3(str2double(get(handles.taustart,'String')),str2double(get(handles.taufinish,'String')),str2double(get(handles.framerate,'String')),handles.I,handles.images,handles.averageBeat,handles.outlier,str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String')),get(handles.tfilt,'Value'),str2double(get(handles.r2cut,'String')))
            map[np.isnan[map]] = 0
            cla
            axes(handles.mapaxes)
            imshow(map,[],'InitialMagnification',800)
            plt.title('Relaxation Constant Map')
            pretty = get(handles.colmap,'String')
            jetcolormap = (colormap(pretty[get(handles.colmap,'Value')]))
            jetcolormap[1,:] = np.array([1,1,1])
            colormap(jetcolormap)
            if get(handles.apdscale,'Value') == 1:
                caxis(np.array([int(np.floor(np.amin(np.amin(alll)))),np.ceil(np.amax(np.amax(alll)))]))
            if get(handles.apdscale,'Value') == 2:
                caxis(np.array([str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String'))]))
            freezeColors
            # colorbar
            axes(handles.cb)
            cla('reset')
            hcb = colorbar
            hcb.Location = 'southoutside'
            cpos = hcb.Position
            cpos[4] = 4 * cpos(4)
            hcb.Position = cpos
            if get(handles.apdscale,'Value') == 1:
                np.amax(np.amax(alll))
                np.amin(np.amin(alll))
                stepp = (np.ceil(np.amax(np.amax(alll))) - int(np.floor(np.amin(np.amin(alll))))) / 5
                hcb.TickLabels = np.array([np.arange(int(np.floor(np.amin(np.amin(alll)))),np.ceil(np.amax(np.amax(alll)))+stepp,stepp)])
                hcb.Ticks = np.array([0.01,np.arange(0.2,1+0.2,0.2)])
                hcb.Label.String = 'Relaxation Constant (ms)'
                plt.axis('off')
            if get(handles.apdscale,'Value') == 2:
                stepp = (str2double(get(handles.cmax,'String')) - str2double(get(handles.cmin,'String'))) / 5
                hcb.TickLabels = np.array([np.arange(str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String'))+stepp,stepp)])
                hcb.Ticks = np.array([0.01,np.arange(0.2,1+0.2,0.2)])
                hcb.Label.String = 'Relaxation Constant (ms)'
                plt.axis('off')
            os.delete(wb)
            taus = alll
            tausp = prctile(taus,np.array([5,50,95]))
            handles.rdata[4,1] = mean(taus)
            handles.rdata[4,2] = std(taus)
            handles.rdata[4,3] = std(taus) / np.asarray(taus).size
            handles.rdata[4,4] = std(taus) * std(taus)
            handles.rdata[4,4] = std(taus) * std(taus)
            handles.rdata[4,5] = ((tausp(3) - tausp(1)) / tausp(2))
            rownames = get(handles.rtable,'RowName')
            rownames[4] = 'Tau'
            set(handles.rtable,'RowName',rownames)
            set(handles.rtable,'data',handles.rdata)
        
        if choice == 9:
            map = fluo_map(str2double(get(handles.framerate,'String')),handles.I,handles.images,get(handles.tfilt,'Value'),handles.averageBeat)
            map[np.isnan[map]] = 0
            alll = map(map > 0)
            cla
            axes(handles.mapaxes)
            if get(handles.apdscale,'Value') == 1:
                imshow(map,np.array([int(np.floor(np.amin(np.amin(alll)))),np.ceil(np.amax(np.amax(alll)))]),'InitialMagnification',800)
            if get(handles.apdscale,'Value') == 2:
                imshow(map,np.array([str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String'))]),'InitialMagnification',800)
            plt.title('Amplitude Map')
            pretty = get(handles.colmap,'String')
            jetcolormap = (colormap(pretty[get(handles.colmap,'Value')]))
            jetcolormap[1,:] = np.array([1,1,1])
            colormap(jetcolormap)
            freezeColors
            # colorbar
            axes(handles.cb)
            cla('reset')
            hcb = colorbar
            hcb.Location = 'southoutside'
            cpos = hcb.Position
            cpos[4] = 4 * cpos(4)
            hcb.Position = cpos
            if get(handles.apdscale,'Value') == 1:
                stepp = (np.ceil(np.amax(np.amax(alll))) - int(np.floor(np.amin(np.amin(alll))))) / 5
                hcb.TickLabels = np.array([np.arange(int(np.floor(np.amin(np.amin(alll)))),np.ceil(np.amax(np.amax(alll)))+stepp,stepp)])
                hcb.Ticks = np.array([0.01,np.arange(0.2,1+0.2,0.2)])
                hcb.Label.String = 'Relaxation Constant (ms)'
                plt.axis('off')
            if get(handles.apdscale,'Value') == 2:
                stepp = (str2double(get(handles.cmax,'String')) - str2double(get(handles.cmin,'String'))) / 5
                hcb.TickLabels = np.array([np.arange(str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String'))+stepp,stepp)])
                hcb.Ticks = np.array([0.01,np.arange(0.2,1+0.2,0.2)])
                hcb.Label.String = 'Relaxation Constant (ms)'
                plt.axis('off')
            hcb.Label.String = 'Signal Level'
            plt.axis('off')
            handles.rdata[4,1] = NaN
            handles.rdata[4,2] = NaN
            handles.rdata[4,3] = NaN
            handles.rdata[4,4] = NaN
            handles.rdata[4,4] = NaN
            handles.rdata[4,5] = NaN
            rownames = get(handles.rtable,'RowName')
            rownames[4] = ''
            set(handles.meanchange,'String',rownames[4])
            set(handles.rtable,'data',handles.rdata)
        
        # Draw Contours
        drawcon = handles.drawcon
        if drawcon == 1:
            axes(handles.mapaxes)
            hold('on')
            for j in np.arange(0,np.amax(np.amax(map)) + str2double(handles.conbon)+str2double(handles.conbon),str2double(handles.conbon)).reshape(-1):
                mapmask = (map <= j)
                A = np.multiply(map,mapmask)
                cons = bwboundaries(A)
                for i in np.arange(1,cons.shape[1-1]+1).reshape(-1):
                    plt.plot(cons[i][:, 1], cons[i][:, 0], 'k', linewidth=1)
        
        #    handles.mask=handles.hold_mask; #keep and reinsate overall mask at end
        handles.holdmap = map
        handles.holdcvmap = CVmap
        drawnow()
        guidata(hObject,handles)
        drawnow()
        # Hints: contents = cellstr(get(hObject,'String')) returns apdvaluechoice contents as cell array
    #        contents{get(hObject,'Value')} returns selected item from apdvaluechoice
        
        # --- Executes during object creation, after setting all properties.
        
    def Mapchoice_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on selection change in mapoptions.
        
    # def mapoptions_Callback(*argsne,*argsne,*argsne): 
    #     pass
        # hObject    handle to mapoptions (see GCBO)
    # eventdata  reserved - to be defined in a future version of MATLAB
    # handles    structure with handles and user data (see GUIDATA)
        
        # Hints: contents = cellstr(get(hObject,'String')) returns mapoptions contents as cell array
    #        contents{get(hObject,'Value')} returns selected item from mapoptions
        
        # --- Executes during object creation, after setting all properties.
        
    def mapoptions_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on button press in ExportMap.
        
    def ExportMap_Callback(*argsne,**kwargs): 
        GUI_fig_children = get(gcf,'children')
        Fig_Axes = findobj(GUI_fig_children,'type','Axes')
        fig = figure
        ax = axes
        clf
        new_handle = copyobj(handles.mapaxes,fig)
        pretty = get(handles.colmap,'String')
        jetcolormap = (colormap(pretty[get(handles.colmap,'Value')]))
        jetcolormap[1,:] = np.array([1,1,1])
        colormap(jetcolormap)
        set(gca,'ActivePositionProperty','outerposition')
        set(gca,'Units','normalized')
        set(gca,'OuterPosition',np.array([0,0,1,1]))
        set(gca,'position',np.array([0.13,0.11,0.775,0.815]))
        # --- Executes on button press in actpoints.
        
    def actpoints_Callback(hObject = None,*argsne,**kwargs): 
        # hObject    handle to actpoints (see GCBO)
    # eventdata  reserved - to be defined in a future version of MATLAB
    # handles    structure with handles and user data (see GUIDATA)
        # handles = guidata(hObject)
        # if get(handles.actfittimes,'Value') == 1:
        #     *argsx,act_y,act_t,*argsmap(str2double(get(handles.pixelsize,'String')),str2double(get(handles.framerate,'String')),handles.cvimages,handles.mask,get(handles.velout,'Value'),str2double(get(handles.minvel,'String')),str2double(get(handles.maxvel,'String')),get(handles.velalgo,'Value'),0,100,str2double(get(handles.winsize,'String')),str2double(get(handles.beforeGUI,'String')),str2double(get(handles.wint,'String')),0,str2double(get(handles.t,'String')),get(handles.tfilt,'Value'),get(handles.usespline,'Value'),str2double(get(handles.splineN,'String')))
        
        # if get(handles.actfittimes,'Value') == 2:
        #     *argsx,act_y,act_t,*argsmap(str2double(get(handles.pixelsize,'String')),str2double(get(handles.framerate,'String')),handles.cvimages,handles.mask,get(handles.velout,'Value'),str2double(get(handles.minvel,'String')),str2double(get(handles.maxvel,'String')),get(handles.velalgo,'Value'),str2double(get(handles.MINt,'String')),str2double(get(handles.MAXt,'String')),str2double(get(handles.winsize,'String')),str2double(get(handles.beforeGUI,'String')),str2double(get(handles.wint,'String')),0,str2double(get(handles.t,'String')),get(handles.tfilt,'Value'),get(handles.usespline,'Value'),str2double(get(handles.splineN,'String')))
        
        figure
        hold('on')
        plot3(act_x,act_y,act_t,'.k')
        plt.title('activation points')
        plt.zlabel('time (ms)','FontSize',20)
        plt.xlabel('x','FontSize',20)
        plt.ylabel('y','FontSize',20)
        plt.zlim(np.array([0,15]))
        hold('off')
        # --- Executes on button press in velhist.
        
    def velhist_Callback(hObject = None,*argsne,**kwargs): 
        # hObject    handle to velhist (see GCBO)
    # eventdata  reserved - to be defined in a future version of MATLAB
    # handles    structure with handles and user data (see GUIDATA)
        handles = guidata(hObject)
        if get(handles.actfittimes,'Value') == 1:
            pass
        #     *args,*argsmap(str2double(get(handles.pixelsize,'String')),str2double(get(handles.framerate,'String')),handles.cvimages,handles.mask,get(handles.velout,'Value'),str2double(get(handles.minvel,'String')),str2double(get(handles.maxvel,'String')),get(handles.velalgo,'Value'),0,100,str2double(get(handles.winsize,'String')),str2double(get(handles.beforeGUI,'String')),str2double(get(handles.wint,'String')),0,str2double(get(handles.t,'String')),get(handles.tfilt,'Value'),get(handles.usespline,'Value'),str2double(get(handles.splineN,'String')))
        
        # if get(handles.actfittimes,'Value') == 2:
        #     *args,*argsmap(str2double(get(handles.pixelsize,'String')),str2double(get(handles.framerate,'String')),handles.cvimages,handles.mask,get(handles.velout,'Value'),str2double(get(handles.minvel,'String')),str2double(get(handles.maxvel,'String')),get(handles.velalgo,'Value'),str2double(get(handles.MINt,'String')),str2double(get(handles.MAXt,'String')),str2double(get(handles.winsize,'String')),str2double(get(handles.beforeGUI,'String')),str2double(get(handles.wint,'String')),0,str2double(get(handles.t,'String')),get(handles.tfilt,'Value'),get(handles.usespline,'Value'),str2double(get(handles.splineN,'String')))
        
        # figure
        # histogram(vout,str2double(get(handles.binnumber,'String')))
        # hold('on')
        # #line([mean(vout) mean(vout)],[0 300],'Color','r','Linewidth',3)
        # plt.xlabel('Conduction Velocity (cm/s)')
        # plt.ylabel('Number of Pixels')
        # hold('off')
        # --- Executes on button press in APDdist.
        
    def APDdist_Callback(hObject = None,*argsne,**kwargs): 
        # hObject    handle to APDdist (see GCBO)
    # eventdata  reserved - to be defined in a future version of MATLAB
    # handles    structure with handles and user data (see GUIDATA)
        handles = guidata(hObject)
        alll = handles.apalll
        figure
        histogram(alll,str2double(get(handles.binnumber,'String')))
        hold('on')
        plt.xlabel('Action Potential Duration (ms)')
        plt.ylabel('Number of Pixels')
        hold('off')
        # --- Executes on selection change in apdout.
        
    def apdout_Callback(hObject = None,eventdata = None,*argsne): 
        # hObject    handle to apdout (see GCBO)
    # eventdata  reserved - to be defined in a future version of MATLAB
    # handles    structure with handles and user data (see GUIDATA)
        handles = guidata(hObject)
        Mapchoice_Callback(hObject,eventdata,handles)
        # --- Executes during object creation, after setting all properties.
        
    def apdout_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    def cmin_Callback(hObject = None,eventdata = None,handles = None): 
        pass
        # --- Executes during object creation, after setting all properties.
        
    def cmin_CreateFcn(hObject = None,eventdata = None,handles = None): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    def cmax_Callback(hObject = None,eventdata = None,handles = None): 
        pass
        # --- Executes during object creation, after setting all properties.
        
    def cmax_CreateFcn(hObject = None,eventdata = None,handles = None): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on selection change in velout.
        
    def velout_Callback(*argsne,**kwargs): 
        # hObject    handle to velout (see GCBO)
    # eventdata  reserved - to be defined in a future version of MATLAB
    # handles    structure with handles and user data (see GUIDATA)
        if get(handles.velout,'Value') == 9:
            set(handles.minvel,'String','0.2')
            set(handles.maxvel,'String','0.8')
            set(handles.text64,'String','(0-1)')
        else:
            if get(handles.velout,'Value') == 2:
                set(handles.minvel,'String','0')
                set(handles.maxvel,'String','100')
                set(handles.text64,'String','cm/s')
        
        # Hints: contents = cellstr(get(hObject,'String')) returns velout contents as cell array
    #        contents{get(hObject,'Value')} returns selected item from velout
        
        # --- Executes during object creation, after setting all properties.
        
    def velout_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    # def minvel_Callback(*argsne,): 
    #     pass*argsne,*argsne
    #     # hObject    handle to minvel (see GCBO)
    # eventdata  reserved - to be defined in a future version of MATLAB
    # handles    structure with handles and user data (see GUIDATA)
        
        # Hints: get(hObject,'String') returns contents of minvel as text
    #        str2double(get(hObject,'String')) returns contents of minvel as a double
        
        # --- Executes during object creation, after setting all properties.
        
    def minvel_CreateFcn(hObject = None,*argsne,**kwargs): 
        
        # hObject    handle to minvel (see GCBO)
    # eventdata  reserved - to be defined in a future version of MATLAB
    # handles    empty - handles not created until after all CreateFcns called
        
        # Hint: edit controls usually have a white background on Windows.
    #       See ISPC and COMPUTER.
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    # def maxvel_Callback(*argsne,*argsne,*argsne): 
    #     pass
    #     # hObject    handle to maxvel (see GCBO)
    # # eventdata  reserved - to be defined in a future version of MATLAB
    # # handles    structure with handles and user data (see GUIDATA)
        
    #     # Hints: get(hObject,'String') returns contents of maxvel as text
    # #        str2double(get(hObject,'String')) returns contents of maxvel as a double
        
    #     # --- Executes during object creation, after setting all properties.
        
    # def maxvel_CreateFcn(hObject = None,*argsne,*argsne): 
        # pass
        # hObject    handle to maxvel (see GCBO)
    # eventdata  reserved - to be defined in a future version of MATLAB
    # handles    empty - handles not created until after all CreateFcns called
        
        # Hint: edit controls usually have a white background on Windows.
    #       See ISPC and COMPUTER.
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on button press in pushrecal.
        
    def pushrecal_Callback(hObject = None,eventdata = None,*argsne): 
        # hObject    handle to pushrecal (see GCBO)
    # eventdata  reserved - to be defined in a future version of MATLAB
    # handles    structure with handles and user data (see GUIDATA)
        handles = guidata(hObject)
        contents = cellstr(get(handles.Mapchoice,'String'))
        choice = get(handles.Mapchoice,'Value')
        axes(handles.mapaxes)
        handles.outlier = get(handles.apdout,'Value')
        handles.outliervel = get(handles.velout,'Value')
        wb = waitbar(0.4,'Calculating APD')
        t = str2double(get(handles.t,'String'))
        apmap,meanapd,alll,onedev = mapsbaby(get(handles.aptime1,'Value'),str2double(get(handles.framerate,'String')),t,handles.I,handles.images,handles.averageBeat,handles.outlier,str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String')),get(handles.tfilt,'Value'),str2double(get(handles.beforeGUI,'String')),get(handles.apdbl,'Value'),str2double(get(handles.apdblnum,'String')),handles.medifilt)
        stdall = std(alll)
        palll = prctile(alll,np.array([5,50,95]))
        handles.rdata[4,1] = mean(alll)
        handles.rdata[4,2] = stdall
        handles.rdata[4,3] = stdall / np.sqrt(np.asarray(alll).size)
        handles.rdata[4,4] = stdall * stdall
        handles.rdata[4,5] = ((palll(3) - palll(1)) / palll(2))
        waitbar(0.6,wb,'Calculating conduction velocity')
        # if get(handles.actfittimes,'Value') == 1:
        #     actmap,*argst,quivers_X,quivers_Y,quivers_vx,quivers_vy,*args,quivers_Xout,quivers_Yout,quivers_vxout,quivers_vyout,onedevcv,varicv,SEcv = cvmap(str2double(get(handles.pixelsize,'String')),str2double(get(handles.framerate,'String')),handles.cvimages,handles.mask,get(handles.velout,'Value'),str2double(get(handles.minvel,'String')),str2double(get(handles.maxvel,'String')),get(handles.velalgo,'Value'),0,100,str2double(get(handles.winsize,'String')),str2double(get(handles.beforeGUI,'String')),str2double(get(handles.wint,'String')),0,str2double(get(handles.t,'String')),get(handles.tfilt,'Value'),get(handles.usespline,'Value'),str2double(get(handles.splineN,'String')))
        
        # if get(handles.actfittimes,'Value') == 2:
        #     actmap,*argst,quivers_X,quivers_Y,quivers_vx,quivers_vy,*args,quivers_Xout,quivers_Yout,quivers_vxout,quivers_vyout,onedevcv,varicv,SEcv = cvmap(str2double(get(handles.pixelsize,'String')),str2double(get(handles.framerate,'String')),handles.cvimages,handles.mask,get(handles.velout,'Value'),str2double(get(handles.minvel,'String')),str2double(get(handles.maxvel,'String')),get(handles.velalgo,'Value'),str2double(get(handles.MINt,'String')),str2double(get(handles.MAXt,'String')),str2double(get(handles.winsize,'String')),str2double(get(handles.beforeGUI,'String')),str2double(get(handles.wint,'String')),0,str2double(get(handles.t,'String')),get(handles.tfilt,'Value'),get(handles.usespline,'Value'),str2double(get(handles.splineN,'String')))
        
        #save AP,CV maps and vectors
        handles.apdmap = apmap
        handles.apalll = alll
        handles.actmap = actmap
        handles.act_t = act_t
        handles.vout = vout
        handles.quivers_X = quivers_X
        handles.quivers_Y = quivers_Y
        handles.quivers_vx = quivers_vx
        handles.quivers_vy = quivers_vy
        handles.quivers_Xout = quivers_Xout
        handles.quivers_Yout = quivers_Yout
        handles.quivers_vxout = quivers_vxout
        handles.quivers_vyout = quivers_vyout
        ## update rtable
        rdata = handles.rdata
        APp = prctile(alll,np.array([5,50,95]))
        CVp = prctile(vout,np.array([5,50,95]))
        handles.errCV = np.array([onedevcv,SEcv,varicv,((CVp(3) - CVp(1)) / CVp(2))])
        guidata(hObject,handles)
        rdata[1,1] = mean(alll)
        rdata[1,2] = handles.err(1)
        rdata[1,3] = handles.err(2)
        rdata[1,4] = handles.err(3)
        rdata[1,5] = handles.err(4)
        rdata[2,1] = mean(vout)
        rdata[2,2] = handles.errCV(1)
        rdata[2,3] = handles.errCV(2)
        rdata[2,4] = handles.errCV(3)
        rdata[2,5] = handles.errCV(4)
        handles.rdata = rdata
        set(handles.rtable,'data',handles.rdata)
        #activation time
        tim = act_t
        tim = tim - np.amin(tim)
        allpts = np.asarray(tim).size
        xbins = np.arange(0,np.amax(tim)+0.01,0.01)
        tissueact = 100 * cumsum(hist(tim,xbins)) / allpts
        actmax = str2double(get(handles.actmax,'String'))
        actmin = str2double(get(handles.actmin,'String'))
        Imax = find(tissueact > actmax)
        Imin = find(tissueact > actmin)
        if actmax < 100:
            Imax = Imax(1)
        else:
            Imax = np.amax(tim)
        
        Imin = Imin(1)
        if actmax < 100:
            timmax = Imax * 0.01
        else:
            timmax = Imax
        
        timmin = Imin * 0.01
        timdiff = timmax - timmin
        if actmin == 0:
            timdiff = timmax
        
        if get(handles.checkbox8,'Value') == 1:
            pixar = str2double(get(handles.pixelsize,'String'))
            pixar = pixar * pixar
            pixar = pixar / 1000000
            normfac = pixar * allpts
            timdiff = timdiff / normfac
        
        set(handles.actquote,'String',np.array([num2str(timdiff),' ms']))
        os.delete(wb)
        #maps
        Mapchoice_Callback(hObject,eventdata,handles)
        # hold off
    #scales the values to work with out camera frame rate and resolution
    #factor = pix/exposure*100; # converts to cm/sec
    #CV = mean(v)*factor;
    # disp(['mean_cv: ', num2str(CV), 'cm/sec']);
        
        # text(0,1,['CV: ',num2str(CV), 'cm/sec'], 'Units', 'Normalized')
        
        guidata(hObject,handles)
        # --- Executes on button press in pushmapapply.
        
    def pushmapapply_Callback(hObject = None,eventdata = None,handles = None): 
        # hObject    handle to pushmapapply (see GCBO)
    # eventdata  reserved - to be defined in a future version of MATLAB
    # handles    structure with handles and user data (see GUIDATA)
        Mapchoice_Callback(hObject,eventdata,handles)
        # --- Executes on button press in producemaps.
        
    def producemaps_Callback(hObject = None,eventdata = None,*argsne): 
        # hObject    handle to producemaps (see GCBO)
    # eventdata  reserved - to be defined in a future version of MATLAB
    # handles    structure with handles and user data (see GUIDATA)
        
        handles = guidata(hObject)
        #handles.filming=0;
    ## Store each peak into array
        before = str2double(get(handles.beforeGUI,'String'))
        after = str2double(get(handles.afterGUI,'String'))
        exposure = 1 / str2double(get(handles.framerate,'String'))
        before = np.round(before / exposure)
        
        after = np.round(after / exposure)
        if handles.filming == 0:
            wb = waitbar(0.1,'Preparing Images')
            section_choice = get(handles.listbox2,'Value')
        
        handles.filming
        if handles.filming == 1:
            handles.filmcount = handles.filmcount + 1
            section_choice = handles.filmcount
            section_choice
        
        if str(handles.section[section_choice]) == str('Zoomed Section') == 1 and handles.filming != 1:
            sig = handles.averages(np.arange(handles.q2locs(section_choice,1),handles.q2locs(section_choice,2)+1))
            #Peak Parameters
            maxfluo = np.amax(handles.averages)
            mo = mode(handles.averages)
            handles.peakheight = maxfluo * str2double(get(handles.peakhigh,'String'))
            minpeakdist = str2double(get(handles.minpeak,'String'))
            minpeakdist = np.ceil(minpeakdist / (1 / str2double(get(handles.framerate,'String'))))
            # find peaks
            # *argsfindpeaks(sig,'MINPEAKHEIGHT',handles.peakheight,'MINPEAKDISTANCE',minpeakdist)
            # m
            m = m + handles.q2locs(section_choice,1) - 1
        else:
            m = handles.q2locs[section_choice, :]
        
        f = m[m != 0]
        peaks = len(f)

        ## OVERLAYING ALL BEATS TO MAKE AN AVERAGE BEAT
        # total action potential duration
        APtime = before + after
        # *args = handles.images.shape
        # create empty matrix to fill later on
        if APtime <= num:
            overlay = np.zeros((handles.im.shape[0], handles.im.shape[1], APtime))
        else:
            overlay = np.zeros((handles.im.shape[0], handles.im.shape[1], num))
        # skip the first and last AP to forgo any possible errors exceeding matrix
    # dimensions
        if f(1) <= before:
            startloc = 2
        else:
            startloc = 1
        
        if f(end()) + after > num:
            endloc = np.asarray(f).size - 1
        else:
            endloc = np.asarray(f).size
        
        locRange = np.arange(startloc,endloc+1)
        if len(handles.q2locs) > 1:
            pass
        
        # fill matrix
        if len(locRange) == 0:
            errordlg('Peak too close to start/end of file to be analysed with current window settings, next peak analysed')

        if f[locRange[-1]] + after > len(handles.images[0, 0, :]):
            locRange = locRange[np.arange(1, len(locRange) - 1 + 1)]

        wsmat = []
        tic()
        f1 = f[locRange]
        f1start = (f1[0] - before)
        f1end = (f1[-1] - after)
        handles.imagerange = handles.images[:, :, np.arange(f1start, f1end + 1)]
        if len(locRange) > 1:
            wsmat = np.zeros((handles.images.shape[0], handles.images.shape[1], np.asarray(np.arange(-before, after + 1)).size, np.asarray(locRange).size))
            for x in np.arange(-before, after + 1):
                if f[locRange] + after < len(handles.images[0, 0, :]):
                    overlay[:, :, x + before + 1] = np.sum(handles.images[:, :, f[locRange] + x], axis=2) / len(f)
                    overlay[:, :, x + before + 1] = overlay[:, :, x + before + 1] * np.double(handles.mask)
                    wsmat[:, :, x + before + 1, locRange] = handles.images[:, :, f[locRange] + x]

        
        handles.wsmat = wsmat
        toc
        if len(locRange) == 1:
            for x in np.arange(-before, after+1):
                overlay[:, :, x + before + 1] = handles.images[:, :, f[locRange] + x]
                overlay[:, :, x + before + 1] = overlay[:, :, x + before + 1] * np.double(handles.mask)
        
        handles.cvimages = overlay
        inversion = get(handles.invertopt,'Value')
        ## WRITE TO TIFF STACK
    # # normalise
    #cos = 26/11/16 with new BL removal overlay all negative, so min and max
        if handles.numofpeaksoverall > 1:
            minI = np.amin(overlay)
            maxI = np.amax(overlay)
            averageBeat = overlay - minI
            averageBeat = (2 ** 16 - 1) * averageBeat / (maxI)
            #make 16 bit
            handles.averageBeat = uint16(averageBeat)
        
        if handles.numofpeaksoverall == 1:
            print('hi')
            handles.averageBeat = handles.images
            #handles.cvimages=handles.images;
        
        ## Get numbers
        handles.outlier = get(handles.apdout,'Value')
        if handles.filming == 0:
            waitbar(0.4,wb,'Producing APD map')
            t = str2double(get(handles.t,'String'))
            apmap,meanapd,alll,onedev,var,SE = mapsbaby(get(handles.aptime1,'Value'),str2double(get(handles.framerate,'String')),t,handles.I,handles.images,handles.averageBeat,handles.outlier,str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String')),get(handles.tfilt,'Value'),str2double(get(handles.beforeGUI,'String')),get(handles.apdbl,'Value'),str2double(get(handles.apdblnum,'String')),handles.medifilt)
            # waitbar(0.6,wb,'Producing Isochronal map')
            # if get(handles.actfittimes,'Value') == 1:
            #     actmap,*argst,quivers_X,quivers_Y,quivers_vx,quivers_vy,*args,quivers_Xout,quivers_Yout,quivers_vxout,quivers_vyout,onedevCV,varCV,SECV = cvmap(str2double(get(handles.pixelsize,'String')),str2double(get(handles.framerate,'String')),handles.cvimages,handles.mask,get(handles.velout,'Value'),str2double(get(handles.minvel,'String')),str2double(get(handles.maxvel,'String')),get(handles.velalgo,'Value'),0,400,str2double(get(handles.winsize,'String')),str2double(get(handles.beforeGUI,'String')),str2double(get(handles.wint,'String')),0,str2double(get(handles.t,'String')),get(handles.tfilt,'Value'),get(handles.usespline,'Value'),str2double(get(handles.splineN,'String')))
            # if get(handles.actfittimes,'Value') == 2:
            #     actmap,*argst,quivers_X,quivers_Y,quivers_vx,quivers_vy,*args,quivers_Xout,quivers_Yout,quivers_vxout,quivers_vyout,onedevCV,varCV,SECV = cvmap(str2double(get(handles.pixelsize,'String')),str2double(get(handles.framerate,'String')),handles.cvimages,handles.mask,get(handles.velout,'Value'),str2double(get(handles.minvel,'String')),str2double(get(handles.maxvel,'String')),get(handles.velalgo,'Value'),str2double(get(handles.MINt,'String')),str2double(get(handles.MAXt,'String')),str2double(get(handles.winsize,'String')),str2double(get(handles.beforeGUI,'String')),str2double(get(handles.wint,'String')),0,str2double(get(handles.t,'String')),get(handles.tfilt,'Value'),get(handles.usespline,'Value'),str2double(get(handles.splineN,'String')))
            APp = prctile(alll,np.array([5,50,95]))
            CVp = prctile(vout,np.array([5,50,95]))
            handles.err = np.array([onedev,SE,var,((APp(3) - APp(1)) / APp(2))])
            handles.errCV = np.array([onedevCV,SECV,varCV,((CVp(3) - CVp(1)) / CVp(2))])
            rdata = np.zeros((4,4))
            rdata[1,1] = mean(alll)
            rdata[1,2] = handles.err(1)
            rdata[1,3] = handles.err(2)
            rdata[1,4] = handles.err(3)
            rdata[1,5] = handles.err(4)
            rdata[2,1] = mean(vout)
            rdata[2,2] = handles.errCV(1)
            rdata[2,3] = handles.errCV(2)
            rdata[2,4] = handles.errCV(3)
            rdata[2,5] = handles.errCV(4)
            fmap = fluo_map(str2double(get(handles.framerate,'String')),handles.I,handles.images,get(handles.tfilt,'Value'),handles.averageBeat)
            fmap[np.isnan[fmap]] = 0
            salll = fmap(fmap > 0)
            sp = prctile(salll,np.array([5,50,95]))
            rdata[3,1] = mean(salll)
            rdata[3,2] = std(salll)
            rdata[3,3] = std(salll) / np.sqrt(np.asarray(salll).size)
            rdata[3,4] = std(salll) * std(salll)
            rdata[3,5] = ((sp(3) - sp(1)) / sp(2))
            rdata[rdata == 0] = NaN
            handles.rdata = rdata
            set(handles.rtable,'Data',rdata)
            print('2218')
            #save AP,CV maps and vectors
            handles.apdmap = apmap
            handles.apalll = alll
            handles.actmap = actmap
            handles.act_t = act_t
            handles.vout = vout
            handles.quivers_X = quivers_X
            handles.quivers_Y = quivers_Y
            handles.quivers_vx = quivers_vx
            handles.quivers_vy = quivers_vy
            handles.quivers_Xout = quivers_Xout
            handles.quivers_Yout = quivers_Yout
            handles.quivers_vxout = quivers_vxout
            handles.quivers_vyout = quivers_vyout
            #activation time
            tim = act_t
            tim = tim - np.amin(tim)
            allpts = np.asarray(tim).size
            xbins = np.arange(0,np.amax(tim)+0.01,0.01)
            tissueact = 100 * cumsum(hist(tim,xbins)) / allpts
            actmax = str2double(get(handles.actmax,'String'))
            actmin = str2double(get(handles.actmin,'String'))
            Imax = find(tissueact > actmax)
            Imin = find(tissueact > actmin)
            if actmax < 100:
                Imax = Imax(1)
            else:
                Imax = np.amax(tim)
            Imin = Imin(1)
            if actmax < 100:
                timmax = Imax * 0.01
            else:
                timmax = Imax
            timmin = Imin * 0.01
            timdiff = timmax - timmin
            if actmin == 0:
                timdiff = timmax
            if get(handles.checkbox8,'Value') == 1:
                pixar = str2double(get(handles.pixelsize,'String'))
                pixar = pixar * pixar
                pixar = pixar / 1000000
                normfac = pixar * allpts
                timdiff = timdiff / normfac
            set(handles.actquote,'String',np.array([num2str(timdiff),' ms']))
            if str(handles.section[0]) == str('N/A') == 1:
                set(handles.CLdisp,'String',('N/A - only one peak'))
            if str(handles.section[0]) == str('N/A') == 0:
                if str(handles.section[section_choice]) == str('Zoomed Section') == 1:
                    set(handles.CLdisp,'String','N/A - Custom Section')
                else:
                    set(handles.CLdisp,'String',np.array([num2str((handles.avgCL(2,section_choice))),' ms (Frequency = ',num2str(1000 / np.round(handles.avgCL(2,section_choice),- 1)),' Hz)']))
            os.delete(wb)
        
        guidata(hObject,handles)
        axes(handles.mapaxes)
        ## Upadte roi selector
        
        # Make MAPS!!!!!
        Mapchoice_Callback(hObject,eventdata,handles)
        drawnow()
        guidata(hObject,handles)
        # --- Executes on selection change in threshopt.
        
    def threshopt_Callback(hObject = None,*argsne,handles = None): 
        # hObject    handle to threshopt (see GCBO)
    # eventdata  reserved - to be defined in a future version of MATLAB
    # handles    structure with handles and user data (see GUIDATA)
        
        if get(handles.threshopt,'Value') == 1:
            set(handles.manthresh,'Enable','off')
        
        if get(handles.threshopt,'Value') == 2:
            set(handles.manthresh,'Enable','on')
        
        guidata(hObject,handles)
        # Hints: contents = cellstr(get(hObject,'String')) returns threshopt contents as cell array
    #        contents{get(hObject,'Value')} returns selected item from threshopt
        
        # --- Executes during object creation, after setting all properties.
        
    def threshopt_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    def manthresh_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        pushload_Callback(hObject,eventdata,handles)
        # --- Executes during object creation, after setting all properties.
        
    def manthresh_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on button press in cropimage.
        
    def cropimage_Callback(*argsne,**kwargs): 
        pass
        # --- Executes on selection change in imagedisp.
        
    def imagedisp_Callback(hObject = None,*argsne,**kwargs): 
        handles = guidata(hObject)
        imchoice = get(handles.imagedisp,'Value')
        axes(handles.imageaxes)
        cla
        boundaries = bwboundaries(handles.mask)
        if imchoice == 1:
            imshow(handles.frame1,[],'InitialMagnification',400)
            colormap('gray')
            freezeColors
            hold('on')
            for i in np.arange(1, boundaries.shape[0] + 1):
                plt.plot(boundaries[i][:, 1], boundaries[i][:, 0], 'r', linewidth=2)

            plt.hold(False)  # or plt.hold(False) to turn off hold behavior

        
        if imchoice == 2:
            imshow(handles.fluoim,[],'InitialMagnification',400)
            colormap('jet')
            freezeColors
            hold('on')
            for i in np.arange(1, boundaries.shape[0]+1):
                plt.plot(boundaries[i][:, 1], boundaries[i][:, 0], 'k', linewidth=2)

            plt.hold(False)  # Equivalent to hold('off') in MATLAB
        
        axes(handles.mapaxes)
        pretty = get(handles.colmap,'String')
        jetcolormap = (colormap(pretty[get(handles.colmap,'Value')]))
        jetcolormap[1,:] = np.array([1,1,1])
        colormap(jetcolormap)
        guidata(hObject,handles)
        # Hints: contents = cellstr(get(hObject,'String')) returns imagedisp contents as cell array
    #        contents{get(hObject,'Value')} returns selected item from imagedisp
        
        # --- Executes during object creation, after setting all properties.
        
    def imagedisp_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on button press in cropbox.
        
    def cropbox_Callback(*argsne,**kwargs): 
        pass
        # --- Executes on button press in exportvalues.
        
    def exportvalues_Callback(hObject = None,*argsne,**kwargs): 
        handles = guidata(hObject)
        filename,pathname = uiputfile(np.array([['*.csv'],['*.txt'],['*.mat']]),'Save Map values (Isochronal maps with vector overlay can only be saved as .mat files)')
        # *args= os.path.split(filename)[0],os.path.splitext(os.path.split(filename)[1])[0],os.path.splitext(os.path.split(filename)[1])[1]
        file = np.array([pathname,filename])
        choice = get(handles.Mapchoice,'Value')
        map = handles.holdmap
        # mat file save
        if str('.mat') == str(ext) == 1:
            if choice == 1:
                APD_Dist = map
                save(file,'APD_Dist')
            if choice == 2:
                activation_time = map
                save(file,'activation_time')
            if choice == 3:
                activation_time = map
                xpositions = X_pos
                ypositions = Y_pos
                xvelocities = X_vel
                yvelocities = Y_vel
                velocities = total_vel
                save(file,'activation_time','xpositions','ypositions','xvelocities','yvelocities','velocities','fractional_up')
            if choice == 4:
                activation_time = map
                xpositions = Xout_pos
                ypositions = Yout_pos
                xvelocities = Xout_vel
                yvelocities = Yout_vel
                velocities = total_velout
                save(file,'activation_time','xpositions','ypositions','xvelocities','yvelocities','velocities','fractional_up')
            if choice == 6:
                DI = handles.holdmap
                save(file,'DI')
            if choice == 8:
                tau = handles.holdmap
                save(file,'tau')
            if choice == 9:
                # *argst,*args,*argsevCV,varCV,SECV = cvmap(str2double(get(handles.pixelsize,'String')),str2double(get(handles.framerate,'String')),handles.cvimages,handles.mask,get(handles.velout,'Value'),str2double(get(handles.minvel,'String')),str2double(get(handles.maxvel,'String')),get(handles.velalgo,'Value'),0,200,str2double(get(handles.winsize,'String')),str2double(get(handles.beforeGUI,'String')),str2double(get(handles.wint,'String')),0,str2double(get(handles.t,'String')),get(handles.tfilt,'Value'),get(handles.usespline,'Value'),str2double(get(handles.splineN,'String')))
                cvmean = mean(vout)
                CVdev = onedevCV
                save(file,'RT50','t1080','signal_level','gofmap1080','cvmean','CVdev')
                print('done')
        
        # csv and txt files
        if str('.csv') == str(ext) == 1 or str('.txt') == str(ext) == 1:
            cho = questdlg('How would you like to export the values?','Export','Map','List','List')
            if 'Map' == cho:
                T = table(map)
                writetable(T,file,'Delimiter',',','WriteVariableNames',False)
                if choice == 2 or choice == 3 or choice == 4:
                    cvfile = np.array([pathname,'CV_',filename])
                    T2 = table(handles.holdcvmap)
                    writetable(T2,cvfile,'Delimiter',',','WriteVariableNames',False)
            else:
                if 'List' == cho:
                    listmap = reshape(map,np.asarray(map).size,1)
                    listmap = listmap(listmap > 0)
                    listmap = listmap(np.isnan(listmap) == 0)
                    T = table(listmap)
                    writetable(T,file,'Delimiter',',','WriteVariableNames',False)
                    if choice == 2 or choice == 3 or choice == 4:
                        cvfile = np.array([pathname,'CV_',filename])
                        CVmap = handles.holdcvmap
                        listcvmap = reshape(CVmap,np.asarray(CVmap).size,1)
                        listcvmap = listcvmap(listcvmap > 0)
                        listcvmap = listcvmap(np.isnan(listcvmap) == 0)
                        T = table(listcvmap)
                        writetable(T,cvfile,'Delimiter',',','WriteVariableNames',False)
        
        # --- Executes on button press in act_movie.
        
    def act_movie_Callback(hObject = None,*argsne,**kwargs): 
        handles = guidata(hObject)
        meme = 0
        gifsave = questdlg('Save File?','Gif Save','Yes','No','Yes')
        if 'Yes' == gifsave:
            meme = 1
        else:
            if 'No' == gifsave:
                meme = 0
        
        map = handles.actmap
        if meme == 1:
            a,b = uiputfile('*.gif')
            filename = np.array([b,a])
        
        h = figure
        hold('on')
        imshow(map,[],'InitialMagnification',800)
        map = map
        if get(handles.isoopt,'Value') == 1:
            mini = 0
            maxi = np.amax(np.amax(map))
        
        if get(handles.isoopt,'Value') == 2:
            maxi = str2double(get(handles.isomax,'String'))
            mini = str2double(get(handles.isomin,'String'))
        
        maxiall = np.amax(np.amax(map))
        pretty = get(handles.colmap,'String')
        jetcolormap = (colormap(pretty[get(handles.colmap,'Value')]))
        jetcolormap[1,:] = np.array([0,0,0])
        colormap(jetcolormap)
        caxis(np.array([0,maxi]))
        delay = 0.01
        for j in np.arange(1,np.ceil(maxiall)+0.1,0.1).reshape(-1):
            mapmask = (map < j)
            A = np.multiply(map,mapmask)
            imshow(A,np.array([mini,np.ceil(maxi)]),'Colormap',jetcolormap,'InitialMagnification',400)
            if meme == 1:
                frame = getframe(h)
                im = frame2im(frame)
                imind,cm = rgb2ind(im,256)
                # Write to the GIF File
                if j == 1:
                    imwrite(imind,cm,filename,'gif','Loopcount',inf,'DelayTime',delay)
                else:
                    imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',delay)
        
        
    def isomin_Callback(hObject = None,eventdata = None,handles = None): 
        pass
        # --- Executes during object creation, after setting all properties.
        
    def isomin_CreateFcn(hObject = None,eventdata = None,handles = None): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    def isomax_Callback(hObject = None,eventdata = None,handles = None): 
        pass
        # --- Executes during object creation, after setting all properties.
        
    def isomax_CreateFcn(hObject = None,eventdata = None,handles = None): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on selection change in isoopt.
        
    def isoopt_Callback(hObject = None,eventdata = None,handles = None): 
        handles = guidata(hObject)
        Mapchoice_Callback(hObject,eventdata,handles)
        guidata(hObject,handles)
        # --- Executes during object creation, after setting all properties.
        
    def isoopt_CreateFcn(hObject = None,eventdata = None,handles = None): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on button press in applyiso.
        
    def applyiso_Callback(hObject = None,eventdata = None,handles = None): 
        handles = guidata(hObject)
        listbox2_Callback(hObject,eventdata,handles)
        guidata(hObject,handles)
        # --- Executes on button press in pushbutton17.
        
    def pushbutton17_Callback(hObject = None,eventdata = None,handles = None): 
        handles = guidata(hObject)
        axes(handles.mapaxes)
        choice = questdlg('Video or Image?','Thing','Video','Image','Video')
        if 'Video' == choice:
            filename,pathname = uiputfile(np.array(['*.avi']),'Save .avi image video file of currently displayed maps across all sections')
            # *args= os.path.split(filename)[0],os.path.splitext(os.path.split(filename)[1])[0],os.path.splitext(os.path.split(filename)[1])[1]
            file = np.array([pathname,filename])
            #         if isdepolyed == 0
    #         cd(pathname)
    #         end
            handles.filming = 1
            numsec = len(handles.section)
            vidobj = VideoWriter(file)
            vidobj.FrameRate = 1
            open_(vidobj)
            set(gca,'nextplot','replacechildren')
            guidata(hObject,handles)
            #wb=waitbar(0.1,'Producing video file','WindowStyle', 'modal');
            for k in np.arange(1,numsec+1).reshape(-1):
                handles.filmcount = k
                guidata(hObject,handles)
                producemaps_Callback(hObject,eventdata,handles)
                handles = guidata(hObject)
                axes(handles.mapaxes)
                currFrame = getframe
                writeVideo(vidobj,currFrame)
                0.1 + 0.9 * (k / numsec)
                #waitbar((0.1+0.9*(k/numsec)),wb);
            close_(vidobj)
            #delete(wb)
            set(handles.listbox2,'Value',numsec)
            handles.filming = 0
        else:
            if 'Image' == choice:
                handles.filming = 1
                numsec = len(handles.section)
                for i in np.arange(1,numsec+1).reshape(-1):
                    handles.filmcount = i
                    handles.b2bimage = 1
                    guidata(hObject,handles)
                    Mapchoice_Callback(hObject,eventdata,handles)
                    GUI_fig_children = get(gcf,'children')
                    Fig_Axes = findobj(GUI_fig_children,'type','Axes')
                    fig = figure
                    ax = axes
                    clf
                    new_handle = copyobj(handles.mapaxes,fig)
                    pretty = get(handles.colmap,'String')
                    jetcolormap = (colormap(pretty[get(handles.colmap,'Value')]))
                    jetcolormap[1,:] = np.array([1,1,1])
                    colormap(jetcolormap)
                    set(gca,'ActivePositionProperty','outerposition')
                    set(gca,'Units','normalized')
                    set(gca,'OuterPosition',np.array([i / numsec - 0.2,i / numsec - 0.2,i / numsec,i / numsec]))
                    #set(gca,'position',[0.1300 0.1100 0.7750 0.8150])
                    guidata(hObject,handles)
                handles.filming = 0
                guidata(hObject,handles)
        
        # --- Executes on button press in segEP.
        
    # def segEP_Callback(*argsne,*argsne,*argsne): 
    #     pass
    #     # --- Executes on button press in fold.
        
    # def fold_Callback(*argsne,*argsne,*argsne): 
    #     pass
        
    # def framerate_Callback(*argsne,*argsne,*argsne): 
        # pass
        # --- Executes during object creation, after setting all properties.
        
    def framerate_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    def pixelsize_Callback(*argsne,**kwargs): 
        pass
        # --- Executes during object creation, after setting all properties.
        
    def pixelsize_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    def binnumber_Callback(*argsne,**kwargs): 
        pass
        # --- Executes during object creation, after setting all properties.
        
    def binnumber_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on button press in getpixelinfo.
        
    def getpixelinfo_Callback(hObject = None,*argsne,handles = None): 
        # hObject    handle to getpixelinfo (see GCBO)
    # eventdata  reserved - to be defined in a future version of MATLAB
    # handles    structure with handles and user data (see GUIDATA)
        compare
        guidata(hObject,handles)
        # --- Executes on button press in phasemap.
        
    def phasemap_Callback(*argsne,**kwargs): 
        # hObject    handle to phasemap (see GCBO)
    # eventdata  reserved - to be defined in a future version of MATLAB
    # handles    structure with handles and user data (see GUIDATA)
        
        pass
        # --- Executes on button press in compare.
        
    # def compare_Callback(*argsne,*argsne,*argsne): 
    #     pass
    #     # --- Executes on button press in pushbutton23.
        
    # def pushbutton23_Callback(*argsne,*argsne,*argsne): 
    #     pass
        
    # def beforeGUI_Callback(*argsne,*argsne,*argsne): 
        # pass
        # --- Executes during object creation, after setting all properties.
        
    def beforeGUI_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    # def afterGUI_Callback(*argsne,*argsne,*argsne): 
    #     pass
    #     # --- Executes during object creation, after setting all properties.
        
    def afterGUI_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on selection change in velalgo.
        
    def velalgo_Callback(hObject = None,eventdata = None,handles = None): 
        wb = waitbar(0.5,'Producing Isochronal map')
        guidata(hObject,handles)
        # if get(handles.actfittimes,'Value') == 1:
        #     actmap,*argst,quivers_X,quivers_Y,quivers_vx,quivers_vy,*args,quivers_Xout,quivers_Yout,quivers_vxout,quivers_vyout,onedevCV,varCV,SECV = cvmap(str2double(get(handles.pixelsize,'String')),str2double(get(handles.framerate,'String')),handles.cvimages,handles.mask,get(handles.velout,'Value'),str2double(get(handles.minvel,'String')),str2double(get(handles.maxvel,'String')),get(handles.velalgo,'Value'),0,400,str2double(get(handles.winsize,'String')),str2double(get(handles.beforeGUI,'String')),str2double(get(handles.wint,'String')),0,str2double(get(handles.t,'String')),get(handles.tfilt,'Value'),get(handles.usespline,'Value'),str2double(get(handles.splineN,'String')))
        
        # if get(handles.actfittimes,'Value') == 2:
        #     actmap,*argst,quivers_X,quivers_Y,quivers_vx,quivers_vy,*args,quivers_Xout,quivers_Yout,quivers_vxout,quivers_vyout,onedevCV,varCV,SECV = cvmap(str2double(get(handles.pixelsize,'String')),str2double(get(handles.framerate,'String')),handles.cvimages,handles.mask,get(handles.velout,'Value'),str2double(get(handles.minvel,'String')),str2double(get(handles.maxvel,'String')),get(handles.velalgo,'Value'),str2double(get(handles.MINt,'String')),str2double(get(handles.MAXt,'String')),str2double(get(handles.winsize,'String')),str2double(get(handles.beforeGUI,'String')),str2double(get(handles.wint,'String')),0,str2double(get(handles.t,'String')),get(handles.tfilt,'Value'),get(handles.usespline,'Value'),str2double(get(handles.splineN,'String')))
        
        CVp = prctile(vout,np.array([5,50,95]))
        handles.errCV = np.array([onedevCV,SECV,varCV,((CVp(3) - CVp(1)) / CVp(2))])
        rdata = handles.rdata
        rdata[2,1] = mean(vout)
        rdata[2,2] = handles.errCV(1)
        rdata[2,3] = handles.errCV(2)
        rdata[2,4] = handles.errCV(3)
        rdata[2,5] = handles.errCV(4)
        handles.rdata = rdata
        set(handles.rtable,'Data',rdata)
        print('2768')
        #save CV maps and vectors
        
        handles.actmap = actmap
        handles.act_t = act_t
        handles.quivers_X = quivers_X
        handles.quivers_Y = quivers_Y
        handles.quivers_vx = quivers_vx
        handles.quivers_vy = quivers_vy
        handles.quivers_Xout = quivers_Xout
        handles.quivers_Yout = quivers_Yout
        handles.quivers_vxout = quivers_vxout
        handles.quivers_vyout = quivers_vyout
        os.delete(wb)
        guidata(hObject,handles)
        Mapchoice_Callback(hObject,eventdata,handles)
        guidata(hObject,handles)
        # Hints: contents = cellstr(get(hObject,'String')) returns velalgo contents as cell array
    #        contents{get(hObject,'Value')} returns selected item from velalgo
        
        # --- Executes during object creation, after setting all properties.
        
    def velalgo_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on selection change in actopt.
        
    def actopt_Callback(*argsne,**kwargs): 
        pass
        # --- Executes during object creation, after setting all properties.
        
    def actopt_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    def actmin_Callback(hObject = None,*argsne,**kwargs): 
        handles = guidata(hObject)
        tim = handles.act_t
        tim = tim - np.amin(tim)
        allpts = np.asarray(tim).size
        xbins = np.arange(0,np.amax(tim)+0.01,0.01)
        tissueact = 100 * cumsum(hist(tim,xbins)) / allpts
        actmax = str2double(get(handles.actmax,'String'))
        actmin = str2double(get(handles.actmin,'String'))
        Imax = find(tissueact > actmax)
        Imin = find(tissueact > actmin)
        if actmax < 100:
            Imax = Imax(1)
        else:
            Imax = np.amax(tim)
        
        Imin = Imin(1)
        if actmax < 100:
            timmax = Imax * 0.01
        else:
            timmax = Imax
        
        timmin = Imin * 0.01
        timdiff = timmax - timmin
        if actmin == 0:
            timdiff = timmax
        
        if get(handles.checkbox8,'Value') == 1:
            normfac = 225 / allpts
            timdiff = timdiff * normfac
        
        set(handles.actquote,'String',np.array([num2str(timdiff),' ms']))
        guidata(hObject,handles)
        # Hints: get(hObject,'String') returns contents of actmin as text
    #        str2double(get(hObject,'String')) returns contents of actmin as a double
        
        # --- Executes during object creation, after setting all properties.
        
    def actmin_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    def actmax_Callback(hObject = None,*argsne,**kwargs): 
        handles = guidata(hObject)
        tim = handles.act_t
        tim = tim - np.amin(tim)
        allpts = np.asarray(tim).size
        xbins = np.arange(0,np.amax(tim)+0.01,0.01)
        tissueact = 100 * cumsum(hist(tim,xbins)) / allpts
        actmax = str2double(get(handles.actmax,'String'))
        actmin = str2double(get(handles.actmin,'String'))
        Imax = find(tissueact > actmax)
        Imin = find(tissueact > actmin)
        if actmax < 100:
            Imax = Imax(1)
        else:
            Imax = np.amax(tim)
        
        Imin = Imin(1)
        if actmax < 100:
            timmax = Imax * 0.01
        else:
            timmax = Imax
        
        timmin = Imin * 0.01
        timdiff = timmax - timmin
        if actmin == 0:
            timdiff = timmax
        
        if get(handles.checkbox8,'Value') == 1:
            normfac = 225 / allpts
            timdiff = timdiff * normfac
        
        set(handles.actquote,'String',np.array([num2str(timdiff),' ms']))
        guidata(hObject,handles)
        # --- Executes during object creation, after setting all properties.
        
    def actmax_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on button press in checkbox8.
        
    def checkbox8_Callback(**kwargs): 
        handles = guidata(hObject)
        tim = handles.act_t
        tim = tim - np.amin(tim)
        allpts = np.asarray(tim).size
        xbins = np.arange(0,np.amax(tim)+0.01,0.01)
        tissueact = 100 * cumsum(hist(tim,xbins)) / allpts
        actmax = str2double(get(handles.actmax,'String'))
        actmin = str2double(get(handles.actmin,'String'))
        Imax = find(tissueact > actmax)
        Imin = find(tissueact > actmin)
        if actmax < 100:
            Imax = Imax(1)
        else:
            Imax = np.amax(tim)
        
        Imin = Imin(1)
        if actmax < 100:
            timmax = Imax * 0.01
        else:
            timmax = Imax
        
        timmin = Imin * 0.01
        timdiff = timmax - timmin
        if actmin == 0:
            timdiff = timmax
        
        if get(handles.checkbox8,'Value') == 1:
            pixar = str2double(get(handles.pixelsize,'String'))
            pixar = pixar * pixar
            pixar = pixar / 1000000
            normfac = pixar * allpts
            timdiff = timdiff / normfac
        
        set(handles.actquote,'String',np.array([num2str(timdiff),' ms']))
        
    def t_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        t = str2double(get(handles.t,'String'))
        apmap,meanapd,alll,onedev,var,SE = mapsbaby(get(handles.aptime1,'Value'),str2double(get(handles.framerate,'String')),t,handles.I,handles.images,handles.averageBeat,handles.outlier,str2double(get(handles.cmin,'String')),str2double(get(handles.cmax,'String')),get(handles.tfilt,'Value'),str2double(get(handles.beforeGUI,'String')),get(handles.apdbl,'Value'),str2double(get(handles.apdblnum,'String')),handles.medifilt)
        #alll=apmap(apmap>0);
        APp = prctile(alll,np.array([5,50,95]))
        handles.err = np.array([onedev,SE,var,((APp(3) - APp(1)) / APp(2))])
        #save AP,CV maps and vectors
        handles.apdmap = apmap
        handles.apalll = alll
        rdata = handles.rdata
        rdata[1,1] = mean(alll)
        rdata[1,2] = handles.err(1)
        rdata[1,3] = handles.err(2)
        rdata[1,4] = handles.err(3)
        rdata[1,5] = handles.err(4)
        handles.rdata = rdata
        guidata(hObject,handles)
        set(handles.rtable,'Data',rdata)
        print('2977')
        mapcho = get(handles.Mapchoice,'Value')
        if mapcho == 1:
            Mapchoice_Callback(hObject,eventdata,handles)
        
        guidata(hObject,handles)
        # --- Executes during object creation, after setting all properties.
        
    def t_CreateFcn(hObject = None,*argsne,**kwargs): 
        # hObject    handle to t (see GCBO)
    # eventdata  reserved - to be defined in a future version of MATLAB
    # handles    empty - handles not created until after all CreateFcns called
        
        # Hint: edit controls usually have a white background on Windows.
    #       See ISPC and COMPUTER.
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on selection change in apdscale.
        
    def apdscale_Callback(hObject = None,eventdata = None,*argsne): 
        # hObject    handle to apdscale (see GCBO)
    # eventdata  reserved - to be defined in a future version of MATLAB
    # handles    structure with handles and user data (see GUIDATA)
        handles = guidata(hObject)
        Mapchoice_Callback(hObject,eventdata,handles)
        # --- Executes during object creation, after setting all properties.
        
    def apdscale_CreateFcn(hObject = None,*argsne,**kwargs): 
        pass
        # hObject    handle to apdscale (see GCBO)
    # eventdata  reserved - to be defined in a future version of MATLAB
    # handles    empty - handles not created until after all CreateFcns called
        
        # Hint: popupmenu controls usually have a white background on Windows.
    #       See ISPC and COMPUTER.
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    def MINt_Callback(*argsne,**kwargs): 
        pass
        # --- Executes during object creation, after setting all properties.
        
    def MINt_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    def MAXt_Callback(*argsne,**kwargs): 
        pass
        # --- Executes during object creation, after setting all properties.
        
    def MAXt_CreateFcn(hObject = None,eventdata = None,handles = None): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on selection change in actfittimes.
        
    def actfittimes_Callback(*argsne,**kwargs): 
        pass
        # --- Executes during object creation, after setting all properties.
        
    def actfittimes_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    def winsize_Callback(*argsne,**kwargs): 
        pass
        # --- Executes during object creation, after setting all properties.
        
    def winsize_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    def taustart_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        contents = cellstr(get(handles.Mapchoice,'String'))
        choice = get(handles.Mapchoice,'Value')
        if choice == 8:
            Mapchoice_Callback(hObject,eventdata,handles)
        
        # --- Executes during object creation, after setting all properties.
        
    def taustart_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    def r2cut_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        contents = cellstr(get(handles.Mapchoice,'String'))
        choice = get(handles.Mapchoice,'Value')
        if choice == 8:
            Mapchoice_Callback(hObject,eventdata,handles)
        
        # --- Executes during object creation, after setting all properties.
        
    def r2cut_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    def apdblnum_Callback(*argsne,**kwargs): 
        pass
        # --- Executes during object creation, after setting all properties.
        
    def apdblnum_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on selection change in apdbl.
        
    def apdbl_Callback(hObject = None,eventdata = None,handles = None): 
        t_Callback(hObject,eventdata,handles)
        # --- Executes during object creation, after setting all properties.
        
    def apdbl_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    def taufinish_Callback(*argsne,**kwargs): 
        pass
        # --- Executes during object creation, after setting all properties.
        
    def taufinish_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    def peakhigh_Callback(*argsne,**kwargs): 
        pass
        # --- Executes during object creation, after setting all properties.
        
    def peakhigh_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    # def scal_Callback(*argsne,*argsne,*argsne): 
        # pass
        # --- Executes during object creation, after setting all properties.
        
    def scal_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    # def wint_Callback(*argsne,*argsne,*argsne): 
    #     pass
        # --- Executes during object creation, after setting all properties.
        
    def wint_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on button press in altanal.
        
    # def altanal_Callback(hObject = None,*argsne,*argsne): 
    #     pass
    #     # --- Executes on button press in removef.
        
    # def removef_Callback(*argsne,*argsne,*argsne): 
    #     pass
    #     # --- Executes on selection change in aptime1.
        
    def aptime1_Callback(hObject = None,eventdata = None,handles = None): 
        t_Callback(hObject,eventdata,handles)
        # --- Executes during object creation, after setting all properties.
        
    def aptime1_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on button press in configure.
        
    def configure_Callback(*argsne,**kwargs): 
        # Construct a questdlg
        choice = questdlg('Would you like to load a configuration file or save current settings?','Config File','New File','Load Settings','Load Settings')
        # Handle response
        if 'New File' == choice:
            fname,PathName = uiputfile('*.txt')
            filename = np.array([PathName,fname])
            # create file for writing too
            fileID = open(filename,'w')
            dt = datetime('now')
            dt = datestr(dt)
            fileID.write('Configuration File for use in ElectroMap\r\n' % ())
            fileID.write(np.array(['Created: ',dt,'\r\n']) % ())
            fileID.write(np.array(['Notes:\r\n\r\n\r\n']) % ())
            fileID.write(np.array(['--- DO NOT EDIT VARIABLE NAMES OR REMOVE ! BELOW THIS POINT, {} = units, () = settings in ElectroMap or Notes ---\r\n\r\n']) % ())
            #cell arrays for popdown menu strings
            threshopt_string = get(handles.threshopt,'String')
            thershopt_string = threshopt_string[get(handles.threshopt,'Value')]
            sfilt_string = get(handles.sfilt,'String')
            sfilt_string = sfilt_string[get(handles.sfilt,'Value')]
            segchoice_string = get(handles.segchoice,'String')
            segchoice_string = segchoice_string[get(handles.segchoice,'Value')]
            BLopt_string = get(handles.BLopt,'String')
            BLopt_string = BLopt_string[get(handles.BLopt,'Value')]
            tfilt_string = get(handles.tfilt,'String')
            tfilt_string = tfilt_string[get(handles.tfilt,'Value')]
            apdbl_string = get(handles.apdbl,'String')
            apdbl_string = apdbl_string[get(handles.apdbl,'Value')]
            aptime1_string = get(handles.aptime1,'String')
            aptime1_string = aptime1_string[get(handles.aptime1,'Value')]
            velalgo_string = get(handles.velalgo,'String')
            velalgo_string = velalgo_string[get(handles.velalgo,'Value')]
            actfittimes_string = get(handles.actfittimes,'String')
            actfittimes_string = actfittimes_string[get(handles.actfittimes,'Value')]
            velout_string = get(handles.velout,'String')
            velout_string = velout_string[get(handles.velout,'Value')]
            apdout_string = get(handles.apdout,'String')
            apdout_string = apdout_string[get(handles.apdout,'Value')]
            apdscale_string = get(handles.apdscale,'String')
            apdscale_string = apdscale_string[get(handles.apdscale,'Value')]
            # get settings and put into file
            fileID.write(np.array(['framerate=',get(handles.framerate,'String'),'! {kHz} !\r\n']) % ())
            fileID.write(np.array(['pixelsize=',get(handles.pixelsize,'String'),'! {ms} !!\r\n']) % ())
            fileID.write(np.array(['threshopt=',num2str(get(handles.threshopt,'Value')),'! (',thershopt_string,')',' !\r\n']) % ())
            fileID.write(np.array(['manthresh=',get(handles.manthresh,'String'),'! {Percent} (Change from automatically generated threshold) !\r\n']) % ())
            fileID.write(np.array(['sfilt=',num2str(get(handles.sfilt,'Value')),'! (',sfilt_string,') !\r\n']) % ())
            fileID.write(np.array(['sfiltsize=',get(handles.sfiltsize,'String'),'! {Pixels} !\r\n']) % ())
            fileID.write(np.array(['segchoice=',num2str(get(handles.segchoice,'Value')),'! (',segchoice_string,') !\r\n']) % ())
            fileID.write(np.array(['segsize=',get(handles.segsize,'String'),'! !\r\n']) % ())
            fileID.write(np.array(['invertopt=',num2str(get(handles.invertopt,'Value')),'! (1 means invert signal) !\r\n']) % ())
            fileID.write(np.array(['BLopt=',num2str(get(handles.BLopt,'Value')),'! (',BLopt_string,') !\r\n']) % ())
            fileID.write(np.array(['thlen=',get(handles.thlen,'String'),'! {ms}, (Length of top-hat filter) !\r\n']) % ())
            fileID.write(np.array(['tfilt=',num2str(get(handles.tfilt,'Value')),'! (',tfilt_string,') !\r\n']) % ())
            fileID.write(np.array(['minpeak=',get(handles.minpeak,'String'),'! {ms} !\r\n']) % ())
            fileID.write(np.array(['peakhigh=',get(handles.peakhigh,'String'),'! !\r\n']) % ())
            fileID.write(np.array(['minnum=',get(handles.minnum,'String'),'! !\r\n']) % ())
            fileID.write(np.array(['minbound=',get(handles.minbound,'String'),'! {ms} !\r\n']) % ())
            fileID.write(np.array(['beforeGUI=',get(handles.beforeGUI,'String'),'! {ms} !\r\n']) % ())
            fileID.write(np.array(['afterGUI=',get(handles.afterGUI,'String'),'! {ms} !\r\n']) % ())
            fileID.write(np.array(['apdbl=',num2str(get(handles.apdbl,'Value')),'! (',apdbl_string,') !\r\n']) % ())
            fileID.write(np.array(['apdblnum=',get(handles.apdblnum,'String'),'! {ms} !\r\n']) % ())
            fileID.write(np.array(['aptime1=',num2str(get(handles.aptime1,'Value')),'! (',aptime1_string,') !\r\n']) % ())
            fileID.write(np.array(['taustart=',get(handles.taustart,'String'),'! {Percent} !\r\n']) % ())
            fileID.write(np.array(['taufinish=',get(handles.taufinish,'String'),'! {Percent} !\r\n']) % ())
            fileID.write(np.array(['r2cut=',get(handles.r2cut,'String'),'! !\r\n']) % ())
            fileID.write(np.array(['velalgo=',num2str(get(handles.velalgo,'Value')),'! (',velalgo_string,') (Activation Measure) !\r\n']) % ())
            fileID.write(np.array(['isoopt=',num2str(get(handles.isoopt,'Value')),'! !\r\n']) % ())
            fileID.write(np.array(['isomin=',get(handles.isomin,'String'),'! {ms} !\r\n']) % ())
            fileID.write(np.array(['isomax=',get(handles.isomax,'String'),'! {ms} !\r\n']) % ())
            fileID.write(np.array(['actfittimes=',num2str(get(handles.actfittimes,'Value')),'! (',actfittimes_string,') !\r\n']) % ())
            fileID.write(np.array(['MINt=',get(handles.MINt,'String'),'! {ms} (Minimum activation time for multi-vector fit) !\r\n']) % ())
            fileID.write(np.array(['MAXt=',get(handles.MAXt,'String'),'! {ms} (Maximum activation time for multi-vector fit) !\r\n']) % ())
            fileID.write(np.array(['velout=',num2str(get(handles.velout,'Value')),'! (',velout_string,') (Local Velocity outlier removal) !\r\n']) % ())
            fileID.write(np.array(['minvel=',get(handles.minvel,'String'),'! {ms} (Minimum calcualted velocity that is not discarded) !\r\n']) % ())
            fileID.write(np.array(['maxvel=',get(handles.maxvel,'String'),'! {ms} (Maximum calculated velocity that is not discarded) !\r\n']) % ())
            fileID.write(np.array(['winsize=',get(handles.winsize,'String'),'!{Pixels} (Local window size) !\r\n']) % ())
            fileID.write(np.array(['scal=',get(handles.scal,'String'),'! (Size of overlaid velocity vectors) !\r\n']) % ())
            fileID.write(np.array(['wint=',get(handles.wint,'String'),'! (Maximum time diffrence allowed in local window fit) !\r\n']) % ())
            fileID.write(np.array(['apdout=',num2str(get(handles.apdout,'Value')),'! (',apdout_string,') (APD/CaD outlier removal) !\r\n']) % ())
            fileID.write(np.array(['apdscale=',num2str(get(handles.apdscale,'Value')),'! (',apdscale_string,') !\r\n']) % ())
            fileID.write(np.array(['cmin=',get(handles.cmin,'String'),'! {ms} (manual colour map minimum)!\r\n']) % ())
            fileID.write(np.array(['cmax=',get(handles.cmax,'String'),'! {ms} (manual colour map maximum)!\r\n']) % ())
            fileID.write(np.array(['t=',get(handles.t,'String'),'! {Percent} (APD/CaD)!\r\n']) % ())
            fileID.write(np.array(['checkbox8=',num2str(get(handles.checkbox8,'Value')),'! (1 means normalised to {ms/mm2}, 0 absoulte in {ms})!\r\n']) % ())
            fileID.write(np.array(['actmin=',get(handles.actmin,'String'),'! {Percent}!\r\n']) % ())
            fileID.write(np.array(['actmax=',get(handles.actmax,'String'),'! {Percent}!\r\n']) % ())
            fileID.write(np.array(['binnumber=',get(handles.binnumber,'String'),'!!\r\n']) % ())
            # close file
            fileID.close()
        else:
            if 'Load Settings' == choice:
                fname,PathName = uigetfile('*.txt')
                filename = np.array([PathName,fname])
                ## open file for reading
                fid = open(filename,'r','b')
                fstr = np.transpose(fread(fid,'int8=>char'))
                fid.close()
                ## Update GUI
                set(handles.framerate,'String',varEM(fstr,'framerate',0))
                set(handles.pixelsize,'String',varEM(fstr,'pixelsize',0))
                set(handles.threshopt,'Value',varEM(fstr,'threshopt',1))
                set(handles.manthresh,'String',varEM(fstr,'manthresh',0))
                set(handles.sfilt,'Value',varEM(fstr,'sfilt',1))
                set(handles.sfiltsize,'String',varEM(fstr,'sfiltsize',0))
                set(handles.segchoice,'Value',varEM(fstr,'segchoice',1))
                set(handles.segsize,'String',varEM(fstr,'segsize',0))
                set(handles.invertopt,'Value',varEM(fstr,'invertopt',1))
                set(handles.BLopt,'Value',varEM(fstr,'BLopt',1))
                set(handles.tfilt,'Value',varEM(fstr,'tfilt',1))
                set(handles.minpeak,'String',varEM(fstr,'minpeak',0))
                set(handles.peakhigh,'String',varEM(fstr,'peakhigh',0))
                set(handles.minnum,'String',varEM(fstr,'minnum',0))
                set(handles.minbound,'String',varEM(fstr,'minbound',0))
                set(handles.beforeGUI,'String',varEM(fstr,'beforeGUI',0))
                set(handles.afterGUI,'String',varEM(fstr,'afterGUI',0))
                set(handles.apdbl,'Value',varEM(fstr,'apdbl',1))
                set(handles.apdblnum,'String',varEM(fstr,'apdblnum',0))
                set(handles.aptime1,'Value',varEM(fstr,'aptime1',1))
                set(handles.taustart,'String',varEM(fstr,'taustart',0))
                set(handles.taufinish,'String',varEM(fstr,'taufinish',0))
                set(handles.r2cut,'String',varEM(fstr,'r2cut',0))
                set(handles.velalgo,'Value',varEM(fstr,'velalgo',1))
                set(handles.isoopt,'Value',varEM(fstr,'isoopt',1))
                set(handles.isomin,'String',varEM(fstr,'isomin',0))
                set(handles.isomax,'String',varEM(fstr,'isomax',0))
                set(handles.actfittimes,'Value',varEM(fstr,'actfittimes',1))
                set(handles.MINt,'String',varEM(fstr,'MINt',0))
                set(handles.MAXt,'String',varEM(fstr,'MAXt',0))
                set(handles.velout,'Value',varEM(fstr,'velout',1))
                set(handles.minvel,'String',varEM(fstr,'minvel',0))
                set(handles.maxvel,'String',varEM(fstr,'maxvel',0))
                set(handles.winsize,'String',varEM(fstr,'winsize',0))
                set(handles.scal,'String',varEM(fstr,'scal',0))
                set(handles.wint,'String',varEM(fstr,'wint',0))
                set(handles.apdout,'Value',varEM(fstr,'apdout',1))
                set(handles.apdscale,'Value',varEM(fstr,'apdscale',1))
                set(handles.cmin,'String',varEM(fstr,'cmin',0))
                set(handles.cmax,'String',varEM(fstr,'cmax',0))
                set(handles.t,'String',varEM(fstr,'t',0))
                set(handles.checkbox8,'Value',varEM(fstr,'checkbox8',1))
                set(handles.actmin,'String',varEM(fstr,'actmin',0))
                set(handles.actmax,'String',varEM(fstr,'actmax',0))
                set(handles.binnumber,'String',varEM(fstr,'binnumber',0))
                set(handles.thlen,'String',varEM(fstr,'thlen',0))
        
        
    def thlen_Callback(*argsne,**kwargs): 
        pass
        # --- Executes during object creation, after setting all properties.
        
    def thlen_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on selection change in colmap.
        
    def colmap_Callback(hObject = None,eventdata = None,handles = None): 
        handles = guidata(hObject)
        Mapchoice_Callback(hObject,eventdata,handles)
        # --- Executes during object creation, after setting all properties.
        
    def colmap_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on button press in drawcon.
        
    def drawcon_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        Mapchoice_Callback(hObject,eventdata,handles)
        # --- Executes on button press in roibutton.
        
    def roibutton_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        choice = questdlg('Save Current ROI or load previous?','ROI','Save ROI','Load ROI','Save ROI')
        # Handle response
        if 'Save ROI' == choice:
            filename,pathname = uiputfile(np.array(['*.txt']),'Save ROI in text file')
            file = np.array([pathname,filename])
            savemask = handles.mask
            figure
            imshow(savemask,[])
            dlmwrite(file,savemask)
            lmask = maskload(file)
            figure
            imshow(lmask,[])
            savemask.shape
            lmask.shape
        else:
            if 'Load ROI' == choice:
                filename,pathname = uigetfile('*.txt','Select the ROI File')
                file = np.array([pathname,filename])
                handles.loadedmask = maskload(file)
                handles.herefromroiload = 1
                guidata(hObject,handles)
                pushload_Callback(hObject,eventdata,handles)
                handles = guidata(hObject)
                handles.herefromroiload = 0
                guidata(hObject,handles)
        
        guidata(hObject,handles)
        # --- Executes on button press in rawvid.
        
    def rawvid_Callback(hObject = None,*argsne,**kwargs): 
        handles = guidata(hObject)
        axes(handles.imageaxes)
        rows,cols = handles.im.shape
        images = handles.images
        mask = handles.mask
        images = imcomplement(images)
        im = handles.im
        im = double(im)
        im = im - np.amin(np.amin(im))
        im = im / np.amax(np.amax(im))
        im = im * 65535
        im = uint16(im)
        savegif = 1
        if savegif == 1:
            a,b = uiputfile('*.gif')
            filename = np.array([b,a])
        
        prompt = np.array(['Fluorescence threshold (0-1):','Video Start (s):','Video End (s)','Normalise? (0=no, 1=yes)'])
        dlg_title = 'Raw Video Options'
        num_lines = 1
        exposure = 1 / str2double(get(handles.framerate,'String'))
        defaultans = np.array(['0.2','0',num2str(images.shape[3-1] / 1000 * exposure),'1'])
        opts = inputdlg(prompt,dlg_title,num_lines,defaultans)
        flthresh = str2double(opts[0])
        istart = str2double(opts[2])
        iend = str2double(opts[3])
        normF = str2double(opts[4])
        #change is to frame #
        istart = np.round((istart / exposure) * 1000)
        iend = np.round((iend / exposure) * 1000)
        if istart == 0:
            istart = 1
        
        if iend > images.shape[3-1]:
            iend = images.shape[3-1]
        
        for r in np.arange(1, rows + 1):
            for c in np.arange(1, cols + 1):
                sig = np.squeeze(images[r, c, :])
                sig = sig - np.amin(sig)
                sig = np.double(sig)
                if normF == 1:
                    sig = (sig / np.amax(sig)) * 65535
                sig = np.uint16(sig)
                images[r, c, :] = sig
        
        wb = waitbar(0,'Saving Raw Video')
        images = double(images)
        mask = double(mask)
        maxval = np.amax(np.amax(np.amax(images)))
        background = np.matlib.repmat(im,np.array([1,1,3]))
        for i in np.arange(istart, iend + 1):
            waitbar(i / (iend - istart), wb, 'Saving Raw Video')
            combinedImage = background.copy()
            foreground = np.multiply(images[:, :, i], mask)
            foreground = foreground / maxval
            foreground[foreground < flthresh] = 0
            foregroundColourised = colouriseData(foreground, 'j', flthresh, 1)
            c1 = combinedImage[:, :, 0]
            c2 = combinedImage[:, :, 1]
            c3 = combinedImage[:, :, 2]
            f1 = foregroundColourised[:, :, 0]
            f2 = foregroundColourised[:, :, 1]
            f3 = foregroundColourised[:, :, 2]
            c1[np.sum(foreground, axis=2) != 0] = f1[np.sum(foreground, axis=2) != 0]
            c2[np.sum(foreground, axis=2) != 0] = f2[np.sum(foreground, axis=2) != 0]
            c3[np.sum(foreground, axis=2) != 0] = f3[np.sum(foreground, axis=2) != 0]
            combinedImage[:, :, 0] = c1
            combinedImage[:, :, 1] = c2
            combinedImage[:, :, 2] = c3
            plt.hold(True)  # or plt.hold(True) if needed
            plt.axis('image')
            plt.axis('off')
            if savegif == 1:
                delay = 0.01
                imind,cm = rgb2ind(combinedImage,256)
                # Write to the GIF File
                if i == istart:
                    imwrite(imind,cm,filename,'gif','Loopcount',inf,'DelayTime',delay)
                else:
                    imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',delay)
        
        os.delete(wb)
        # --- Executes on button press in resegment.
        
    def resegment_Callback(hObject = None,eventdata = None,handles = None): 
        handles.herefromsegmentpush = 1
        guidata(hObject,handles)
        pushprocess_Callback(hObject,eventdata,handles)
        handles = guidata(hObject)
        handles.herefromsegmentpush = 0
        guidata(hObject,handles)
        producemaps_Callback(hObject,eventdata,handles)
        handles = guidata(hObject)
        guidata(hObject,handles)
        # --- Executes on button press in B2B.
        
    def B2B_Callback(hObject = None,eventdata = None,handles = None): 
        handles.herefromsegmentpush = 1
        set(handles.segsize,'String',1)
        set(handles.segchoice,'Value',2)
        guidata(hObject,handles)
        pushprocess_Callback(hObject,eventdata,handles)
        handles = guidata(hObject)
        handles.herefromsegmentpush = 0
        guidata(hObject,handles)
        producemaps_Callback(hObject,eventdata,handles)
        handles = guidata(hObject)
        guidata(hObject,handles)
        # --- Executes on selection change in winopt.
        
    def winopt_Callback(*argsne,**kwargs): 
        pass
        # --- Executes during object creation, after setting all properties.
        
    def winopt_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on selection change in segsignal.
        
    def segsignal_Callback(hObject = None,eventdata = None,handles = None): 
        handles.herefromsegmentpush = 1
        guidata(hObject,handles)
        pushprocess_Callback(hObject,eventdata,handles)
        handles = guidata(hObject)
        # --- Executes during object creation, after setting all properties.
        
    def segsignal_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    def sfiltsigma_Callback(*argsne,**kwargs): 
        pass
        # --- Executes during object creation, after setting all properties.
        
    def sfiltsigma_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on selection change in usespline.
        
    # def usespline_Callback(*argsne,*argsne,*argsne): 
    #     pass
        # --- Executes during object creation, after setting all properties.
        
    def usespline_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        
    # def splineN_Callback(*argsne,*argsne,*argsne): 
    #     pass
    #     # --- Executes during object creation, after setting all properties.
        
    def splineN_CreateFcn(hObject = None,*argsne,**kwargs): 
        if ispc and get(hObject,'BackgroundColor')==get(0,'defaultUicontrolBackgroundColor'):
            set(hObject,'BackgroundColor','white')
        
        # --- Executes on button press in pushbutton35.
        
    def pushbutton35_Callback(*argsne,**kwargs): 
        pass
        # --------------------------------------------------------------------
        
    def freqmapopt_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        prompt = np.array(['Minimum Frequency (Hz):','Maximum Frequency (Hz):','Frequency Bin Size (Hz)','Window? 0 = no, 1 = hann'])
        dims = np.array([1,35])
        definput = np.array([num2str(handles.fmin),num2str(handles.fmax),num2str(handles.fbin),num2str(handles.dfwin)])
        answer = inputdlg(prompt,'Frequnecy Mapping Options',dims,definput)
        handles.fmin = str2double(answer[0])
        handles.fmax = str2double(answer[2])
        handles.fbin = str2double(answer[3])
        handles.dfwin = str2double(answer[4])
        guidata(hObject,handles)
        if get(handles.Mapchoice,'Value') == 5:
            Mapchoice_Callback(hObject,eventdata,handles)
        
        guidata(hObject,handles)
        # --------------------------------------------------------------------
        
    def ROInum_Callback(hObject = None,*argsne,**kwargs): 
        handles = guidata(hObject)
        prompt = np.array(['Number of ROIs:','Remove overlapping pixels? (0=Yes) (1=No)'])
        dims = np.array([1,35])
        definput = np.array([num2str(handles.roinum),num2str(handles.roisum)])
        answer = inputdlg(prompt,'ROI options',dims,definput)
        handles.roinum = str2double(answer[0])
        handles.roisum = str2double(answer[2])
        guidata(hObject,handles)
        # --------------------------------------------------------------------
        
    # def Untitled_1_Callback(*argsne,*argsne,*argsne): 
    #     pass
    #     # --------------------------------------------------------------------
        
    # def Untitled_5_Callback(*argsne,*argsne,*argsne): 
    #     pass
    #     ## ColourMaps
        
    def coljet_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        set(handles.colmap,'Value',1)
        Mapchoice_Callback(hObject,eventdata,handles)
        guidata(hObject,handles)
        # --------------------------------------------------------------------
        
    def colhsv_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        set(handles.colmap,'Value',2)
        Mapchoice_Callback(hObject,eventdata,handles)
        guidata(hObject,handles)
        # --------------------------------------------------------------------
        
    def colhot_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        set(handles.colmap,'Value',3)
        Mapchoice_Callback(hObject,eventdata,handles)
        guidata(hObject,handles)
        # --------------------------------------------------------------------
        
    def colcool_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        set(handles.colmap,'Value',4)
        Mapchoice_Callback(hObject,eventdata,handles)
        guidata(hObject,handles)
        # --------------------------------------------------------------------
        
    def colparula_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        set(handles.colmap,'Value',5)
        Mapchoice_Callback(hObject,eventdata,handles)
        guidata(hObject,handles)
        # --------------------------------------------------------------------
        
    def colspring_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        set(handles.colmap,'Value',6)
        Mapchoice_Callback(hObject,eventdata,handles)
        guidata(hObject,handles)
        # --------------------------------------------------------------------
        
    def colsummer_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        set(handles.colmap,'Value',7)
        Mapchoice_Callback(hObject,eventdata,handles)
        guidata(hObject,handles)
        # --------------------------------------------------------------------
        
    def colautumn_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        set(handles.colmap,'Value',8)
        Mapchoice_Callback(hObject,eventdata,handles)
        guidata(hObject,handles)
        # --------------------------------------------------------------------
        
    def colwinter_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        set(handles.colmap,'Value',9)
        Mapchoice_Callback(hObject,eventdata,handles)
        guidata(hObject,handles)
        # --------------------------------------------------------------------
        
    # def Untitled_3_Callback(*argsne,*argsne,*argsne): 
    #     pass
    #     # --------------------------------------------------------------------
        
    # def Untitled_4_Callback(*argsne,*argsne,*argsne): 
    #     pass
    #     # --------------------------------------------------------------------
        
    # def Untitled_6_Callback(*argsne,*argsne,*argsne): 
    #     pass
        # --------------------------------------------------------------------
        
    def bgblack_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        handles.bgcol = 'k'
        handles.bgon = 1
        guidata(hObject,handles)
        Mapchoice_Callback(hObject,eventdata,handles)
        guidata(hObject,handles)
        # --------------------------------------------------------------------
        
    def bgwhite_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        handles.bgcol = 'w'
        handles.bgon = 1
        guidata(hObject,handles)
        Mapchoice_Callback(hObject,eventdata,handles)
        guidata(hObject,handles)
        # --------------------------------------------------------------------
        
    def bgtran_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        handles.bgon = 0
        guidata(hObject,handles)
        Mapchoice_Callback(hObject,eventdata,handles)
        guidata(hObject,handles)
        # --------------------------------------------------------------------
        
    def snrcalc_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        prompt = np.array(['Signal before time (from peak) (ms):','Signal after time (from peak) (ms):'])
        dims = np.array([1,35])
        definput = np.array([num2str(handles.snrt1),num2str(handles.snrt2)])
        answer = inputdlg(prompt,'Frequnecy Mapping Options',dims,definput)
        handles.snrt1 = str2double(answer[0])
        handles.snrt2 = str2double(answer[2])
        guidata(hObject,handles)
        if get(handles.Mapchoice,'Value') == 10 or get(handles.Mapchoice,'Value') == 1:
            Mapchoice_Callback(hObject,eventdata,handles)
        
        guidata(hObject,handles)
        # --------------------------------------------------------------------
        
    def ttpset_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        prompt = np.array(['Start Point (%):','End Point (%):'])
        dims = np.array([1,35])
        definput = np.array([num2str(handles.ttpstart),num2str(handles.ttpend)])
        answer = inputdlg(prompt,'Frequnecy Mapping Options',dims,definput)
        handles.ttpstart = str2double(answer[0])
        handles.ttpend = str2double(answer[2])
        guidata(hObject,handles)
        if get(handles.Mapchoice,'Value') == 7:
            Mapchoice_Callback(hObject,eventdata,handles)
        
        guidata(hObject,handles)
        # --------------------------------------------------------------------
        
    # def Untitled_7_Callback(*argsne,*argsne,*argsne): 
    #     pass
    #     # --------------------------------------------------------------------
        
    # def connnnnnnnnnn_Callback(*argsne,*argsne,*argsne): 
    #     pass
        # --------------------------------------------------------------------
        
    def conoff_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        handles.drawcon = 0
        guidata(hObject,handles)
        Mapchoice_Callback(hObject,eventdata,handles)
        # --------------------------------------------------------------------
        
    def conon_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        handles.drawcon = 1
        prompt = np.array(['Contour spacing (map units):'])
        dims = np.array([1,35])
        if len(handles.conbon)==0 == 1:
            handles.framerate
            conbount = 1 / str2double(get(handles.framerate,'String'))
            handles.conbon = num2str(conbount)
        
        definput = np.array([num2str(handles.conbon)])
        answer = inputdlg(prompt,'Contour Setting',dims,definput)
        handles.conbon = answer[0]
        guidata(hObject,handles)
        Mapchoice_Callback(hObject,eventdata,handles)
        # --------------------------------------------------------------------
        
    def nomedifilt_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        handles.medifilt = 0
        guidata(hObject,handles)
        Mapchoice_Callback(hObject,eventdata,handles)
        # --------------------------------------------------------------------
        
    def yesmedifilt_Callback(hObject = None,eventdata = None,*argsne): 
        handles = guidata(hObject)
        handles.medifilt = 1
        guidata(hObject,handles)
        Mapchoice_Callback(hObject,eventdata,handles)
        
    # def squareROI_Callback(*argsne,*argsne,*argsne): 
    #     pass
    #     # --------------------------------------------------------------------
        
    def frameremoval_Callback(hObject = None,eventdata = None,handles = None): 
        # hObject    handle to frameremoval (see GCBO)
    # eventdata  reserved - to be defined in a future version of MATLAB
    # handles    structure with handles and user data (see GUIDATA)
        handles = guidata(hObject)
        handles.drawcon = 1
        prompt = np.array(['Frames before pulse to remove:','Frames after pulse to remove:'])
        dims = np.array([1,35])
        definput = np.array([num2str(handles.pbefore),num2str(handles.pafter)])
        answer = inputdlg(prompt,'Contour Setting',dims,definput)
        handles.pbefore = str2double(answer[0])
        handles.pafter = str2double(answer[2])
        guidata(hObject,handles)
        return varargout
    
    def load_mat_files(self):
            self.mat_files = [os.path.join(self.directory_path, f) for f in os.listdir(self.directory_path) if f.endswith('.mat')]

    def load_image(self):
        mat_file = self.mat_files[self.current_index]
        image_data = scipy.io.loadmat(mat_file)['image_data']
        image = Image.fromarray(image_data)
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep a reference to prevent garbage collection

    def show_next(self):
        if self.current_index < len(self.mat_files) - 1:
            self.current_index += 1
            self.load_image()

    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def main():
        root = tk.Tk()
        app = ImageViewerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
