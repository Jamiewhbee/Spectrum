import numpy as np
import pandas as pd
import os
import re
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.integrate import trapz
import scipy

class Data_macro:
    # Gets inputs, set units, and reads individual files
    def __init__(self,data_path=None,ref_path=None,auto=True,units="cm-1",data_head="none",ref_head="none",as_folder=False,col_names="B",zero_field=False,sep_paths=False,flag_txt=True,high_to_low=None):
        """Initiate instance variables for units, header checks, and Boolean checks
        """
        self.high_to_low = high_to_low
        self.flag_txt = flag_txt
        self.as_folder = as_folder
        self.zero_field = zero_field
        self.col_names = col_names

        # Setting Boolean parameters if none have been entered
        if auto == True:
            self.as_folder == False
            if ref_path == None:
                self.sep_paths = False
                self.zero_field = True
            else:
                self.sep_paths = True

        else:
            self.sep_paths = sep_paths

        if units == None:
            self.units = self.input_units()
        else:
            self.units = units # By default, "units" is set to "cm-1"
        if data_head == None:
            self.data_head = data_head
        else:
            self.data_head = data_head # By default, "data_head" is set to "none" 
        if ref_head == None:
            self.ref_head = self.input_ref_head()
        else:
            self.ref_head = ref_head # By default, "ref_head" is set to "none"
        
        # Depending on the above parameters, get path(s), folder list(s), file list(s), and column names, in that order.
        if data_path == None:
            set_data_path = self.input_path()
        else:
            set_data_path = data_path
        if self.sep_paths == True and ref_path == None:
            set_ref_path = self.input_ref()
        elif self.sep_paths == True:
            set_ref_path = ref_path
        else:
            set_ref_path = set_data_path
        
        if as_folder == False and flag_txt == True:
            if isinstance(set_data_path, str):
                self.data_path = os.path.join(set_data_path, "txt_files")
            else:
                for i in range(0,len(set_data_path)):
                    set_data_path[i] = os.path.join(set_data_path[i], "txt_files")
                self.data_path = set_data_path
            self.ref_path = os.path.join(set_ref_path, "txt_files")
        else:
            self.data_path = set_data_path
            self.ref_path = set_ref_path

        if self.as_folder == True:
            self.folder_list = self.get_folders(self.data_path)
            if self.sep_paths == False:
                self.folder_paths = self.get_subfolders(self.data_path, is_ref=False)
                self.folder_refs = self.get_subfolders(self.data_path, is_ref=True)
            elif self.sep_paths == True:
                folder_paths = []
                for folder in self.folder_list:
                    if self.flag_txt == True:
                        folder_paths.append([os.path.join(self.data_path,folder,"txt_files")])
                    else:
                        folder_paths.append([os.path.join(self.data_path,folder)])
                self.folder_paths = folder_paths
                if self.flag_txt == True:
                    folder_refs = [[os.path.join(self.ref_path,"txt_files")]]
                else:
                    folder_refs = [[self.ref_path]]
                self.folder_refs = folder_refs

    def Auto(self):
        """Loads, processes, and returns a normalised dataframe for a single energy window

        Returns:
            pandas dataframe: Dataframe of normalised spectral data
        """
        if isinstance(self.data_path, str):
            # Same handling as Do_All if the entered path is a string
            Loaded_Data = self.Load_Data(self.data_path)
            Ref_Data = self.Load_Refs(self.ref_path)
            if self.zero_field == True:
                Norm_df = self.Normalise(Loaded_Data, Ref_Data)
            else:
                FileList = self.file_list(self.data_path, is_ref = False)
                cor = False
                for name in FileList:
                    if ".cor" in name:
                        cor = True
                Norm_df = self.Correct(Loaded_Data, Ref_Data, cor)

        else:
            # Normalise each dataset separately, then average
            df_list = []
            for datapath in self.data_path:
                df = self.Load_Data(datapath)
                if self.sep_paths == False:
                    Ref_Data = self.Load_Refs(datapath)
                else:
                    Ref_Data = self.Load_Refs(self.ref_path)
                if self.zero_field == True:
                    df = self.Normalise(df, Ref_Data)
                else:
                    FileList = self.file_list(datapath, is_ref = False)
                    cor = False
                    for name in FileList:
                        if ".cor" in name:
                            cor = True
                    df = self.Correct(df, Ref_Data, cor)   
                df_list.append(df)

            Norm_df = pd.concat(df_list)
            Norm_df = Norm_df.groupby(by=Norm_df.index.name).mean()
            Norm_df = Norm_df[sorted(Norm_df.columns)]

        return Norm_df

    def get_cols(self, list):
        """Extracts the magnetic field or temperature from each file name

        Args:
            list (array): List of file names

        Raises:
            ValueError: Invalid file name format

        Returns:
            array: Array of magnetic field values for each data measurement
        """
        col_vals = []
        for FileName in list:
            if self.col_names == "B":
                col_vals.append(float(re.sub("p",".",FileName[FileName.rfind("a")+1:FileName.rfind("T")])))
            elif self.col_names == "T":
                listints = re.findall(r'\d+',FileName)
                if '0' in listints:
                    listints.remove('0')
                T = int(listints[-1])
                col_vals.append(T)
        if col_vals == []:
            raise ValueError("Unable to read columns from file names. Check that file names fit format.")
        return col_vals

    def get_folders(self, path):
        """Get the relevant folders from the given path

        Returns:
            array: Folder names with matching case/syntax
        """
        folder_list = []
        for folder in os.listdir(path):
            if folder.lower() in ["fir", "mir", "nir", "hmir", "lmir"]:
                folder_list.append(folder)
        if folder_list == []:
            raise ValueError("No folders found in"+self.data_path+". Check that folder names fit format.")
        return folder_list
    
    def get_subfolders(self, path, is_ref):
        """Get list of list of subfolder paths in the case that the measurement and reference data are stored within the same folders

        Args:
            path (string): Path to main folder
            is_ref (bool): Check whether method should look for measurement folders or reference folders

        Returns:
            array: Array of arrays of pathnames for each folder in the main folder
        """
        folder_paths = []
        for folder in self.folder_list:
            subfolder_paths = []
            full_path = os.path.join(path,folder)
            for subfolder in os.listdir(full_path):
                if is_ref == True and "ref" in subfolder.lower():
                    if self.flag_txt == True:
                        subfolder_paths.append(os.path.join(full_path, subfolder,"txt_files"))
                    else:
                        subfolder_paths.append(os.path.join(full_path, subfolder))
                elif is_ref == False and "run" in subfolder.lower() or is_ref == False and "sam" in subfolder.lower():
                    if self.flag_txt == True:
                        subfolder_paths.append(os.path.join(full_path, subfolder,"txt_files"))
                    else:
                        subfolder_paths.append(os.path.join(full_path, subfolder))
            folder_paths.append(subfolder_paths)
        return folder_paths

    def input_path(self):
        """Get path to the measurement folders

        Returns:
            string: Path to folder containing measurement datasets
        """
        data_path = input("Enter the path to the measurement data (or drag and drop): ")
        return data_path
    
    def input_ref(self):
        """Get path to the reference data

        Returns:
            string: Path to folder containing reference datasets
        """
        ref_path = input("Enter the path to the reference data (or drag and drop): ")
        return ref_path
    
    def input_units(self):
        """Get the units for the index. Not called by default unless "units" is set to None when the macro is called

        Returns:
            string: Either "cm-1" or "meV" or "THz"
        """
        units = input("Enter the units to be used for the index (cm-1, meV, or THz): ")
        if units == "cm-1" or units == "meV" or units == "THz":
            return units
        else:
            print("Invalid unit input (must be cm-1, meV, or THz). Using cm-1 units.")
            return "cm-1"
    
    def input_data_head(self):
        """Get method for checking the measurement file headers.
        "all": assume all files have headers, "none": assume no files have headers, "check": check for header (fails if header is a float)

        Returns:
            string: Check
        """
        data_head = input("Enter method for handling measurement file headers (all/none/check): ")
        if data_head == "all" or data_head == "none" or data_head == "check":
            return data_head
        else:
            print("Invalid input: Please select all, none, or check. Using check as default.")
            return "check"
        
    def input_ref_head(self):
        """Get method for checking the reference file headers.
        "all": assume all files have headers, "none": assume no files have headers, "check": check for header (fails if header is a float)

        Returns:
            string: Check
        """
        ref_head = input("Enter method for handling reference file headers (all/none/check): ")
        if ref_head == "all" or ref_head == "none" or ref_head == "check":
            return ref_head
        else:
            print("Invalid input: Please select all, none, or check. Using check as default.")
            return "check"

    def set_units(self, data_ch, is_ref):
        """Sets the units of the dataframe index depending on the user input above

        Args:
            pandas dataframe: Dataframe with corrected units
        """
        if self.units == "cm-1":
            if data_ch.index.name == "Energy (cm$^{-1}$)":
                if is_ref == False:
                    print("Units already in cm-1.")
            elif data_ch.index.name == "Energy (meV)":
                data_ch.index *= 8.0656
                if is_ref == False:
                    print("Units changed from meV to cm-1")
            elif data_ch.index.name == "Energy (THz)":
                data_ch.index *= 33.35641
                if is_ref == False:
                    print("Units changed from THz to cm-1")
            data_ch.index.name = "Energy (cm$^{-1}$)"
            return data_ch
        
        elif self.units == "meV":
            if data_ch.index.name == "Energy (meV)":
                if is_ref == False:
                    print("Units already in meV.")
            elif data_ch.index.name == "Energy (cm$^{-1}$)":
                data_ch.index /= 8.0656
                if is_ref == False:
                    print("Units changed from cm-1 to meV")
            elif data_ch.index.name == "Energy (THz)":
                data_ch.index /= (8.0656 / 33.35641 )
                if is_ref == False:
                    print("Units changed from THz to meV")
            data_ch.index.name = "Energy (meV)"
            return data_ch
        
        elif self.units == "THz":
            if data_ch.index.name == "Energy (THz)":
                if is_ref == False:
                    print("Units already in THz.")
            elif data_ch.index.name == "Energy (cm$^{-1}$)":
                data_ch.index /= 33.35641
                if is_ref == False:
                    print("Units changed from cm-1 to THz")
            elif data_ch.index.name == "Energy (meV)":
                data_ch.index /= (33.35641 / 8.0656)
                if is_ref == False:
                    print("Units changed from meV to THz")
            data_ch.index.name = "Energy (THz)"
            return data_ch
        else:
            print("Unrecognised unit input (must be cm-1, meV, or THz). Assuming cm-1 units.")
            data_ch.index.name = "Energy (cm$^{-1}$)"
            return data_ch

    def file_list(self, pathname, is_ref):
        """Get file list for measurement data and zero-field data

        Args:
            path (str): Path to folder of files
            is_ref (bool): Checks if the files are meant to be measurement data or reference data

        Raises:
            ValueError: Empty measurement folder
            ValueError: Empty reference folder

        Returns:
            array: Sorted list of file names
        """

        sort_files = sorted(os.listdir(pathname))
        # Remove hidden files
        for name in sort_files:
            if name.startswith('.'):
                sort_files.remove(name)
        # Check for empty folder
        if sort_files == []:
            raise ValueError("No files found in",pathname)

        if is_ref == False:
            if self.sep_paths == False and self.zero_field == True:
                return sort_files[2:]
            elif "a00p000T" in sort_files[0]:
                return sort_files[2:]
            else:
                return sort_files
            
        elif is_ref == True:
            if self.sep_paths == False and self.zero_field == True:
                return sort_files[:2]
            elif self.sep_paths == False:
                return sort_files[2:]
            else:
                return sort_files

    # Data-handling methods

    def read_csv(self, path, col, is_ref):
        """Read individual file

        Args:
            path (string): Path to individual file
            col (float): Column name
            is_ref (bool): Check if file is measurement or reference data

        Returns:
            pandas dataframe: Loaded data
        """
        if is_ref == False:
            if self.data_head == "check":
                test_load = pd.read_csv(path, delim_whitespace=True, header=None, index_col=0, nrows = 5)
                if test_load.index.dtype == 'float64':
                    head = None
                else:
                    head = 0
            elif self.data_head == "all":
                head = 0
            elif self.data_head == "none":
                head = None
            return pd.read_csv(path, delim_whitespace=True, header=head, index_col=0, names=[col])

        elif is_ref == True:
            if self.ref_head == "check":
                test_load = pd.read_csv(path, delim_whitespace=True, header=None, index_col=0, nrows = 5)
                if test_load.index.dtype == 'float64':
                    head = None
                else:
                    head = 0
            elif self.ref_head == "all":
                head = 0
            elif self.ref_head == "none":
                head = None
            return pd.read_csv(path, delim_whitespace=True, header=head, index_col=0, names=[col])

    def Load_Data(self, path):
        """Extracts data for the given folder or for the gold reference (if no folder is given). Uses energy as the index and temperature (as an integer) as the column

        Args:
            path (string): Path to data folder

        Returns:
            pandas dataframe: Measured data
        """
        FileList = self.file_list(path, is_ref = False)
        cols = self.get_cols(FileList)
        inconsistencies = ""

        for i in range(0,np.size(FileList)):
            full_path = os.path.join(path, FileList[i])

            if i == 0:
                Loaded_Data = self.read_csv(full_path, col=cols[i], is_ref=False)
            else:
                df = self.read_csv(full_path, col=cols[i], is_ref=False)
                if len(Loaded_Data.index) != len(df.index):
                    inconsistencies = inconsistencies + FileList[i] + ", "
                    Loaded_Data[cols[i]] = np.interp(Loaded_Data.index, df.index, df[cols[i]].values)
                elif np.allclose(Loaded_Data.index,df.index) == False:
                    inconsistencies = inconsistencies + FileList[i] + ", "
                    Loaded_Data[cols[i]] = np.interp(Loaded_Data.index, df.index, df[cols[i]].values)
                else: 
                    Loaded_Data[cols[i]] = df[cols[i]]
        
        if inconsistencies != "":
            print("Inconsistent indices in files "+inconsistencies)
        
        Loaded_Data.index.name = "Energy (cm$^{-1}$)"
        Loaded_Data = self.set_units(data_ch = Loaded_Data, is_ref = False)
        Loaded_Data = Loaded_Data[sorted(cols)]
        return Loaded_Data

    def Load_Refs(self, path):
        """Extracts zero-field data with the index as the energy (in desired units) and the two column headers as the minimum and maximum field (as floats)
                
        Args:
            path (string): Path to data folder

        Returns:
            pandas dataframe: Reference data
        """
        RefList = self.file_list(path, is_ref = True)
        cols = self.get_cols(RefList)
        last_col = sorted(cols)[-1]

        if self.zero_field == True:
            full_path = os.path.join(path,RefList[0])
            Ref_Data = self.read_csv(full_path, col=0, is_ref=True)

            if len(RefList) >= 2: # No interpolation is possible if there is only one zero-field dataset
                full_path = os.path.join(path,RefList[1])
                df = self.read_csv(full_path, col=last_col, is_ref=True)
                Ref_Data[last_col] = df[last_col]

        else:
            inconsistencies = ""
            for i in range(0,np.size(RefList)):
                full_path = os.path.join(path, RefList[i])

                if i == 0:
                    Ref_Data = self.read_csv(full_path, col=cols[i], is_ref=True)
                else:
                    df = self.read_csv(full_path, col=cols[i], is_ref=True)
                    if len(Ref_Data.index) != len(df.index):
                        inconsistencies = inconsistencies + RefList[i] + ", "
                        Ref_Data[cols[i]] = np.interp(Ref_Data.index, df.index, df[cols[i]].values)
                    elif np.allclose(Ref_Data.index,df.index) == False:
                        inconsistencies = inconsistencies + RefList[i] + ", "
                        Ref_Data[cols[i]] = np.interp(Ref_Data.index, df.index, df[cols[i]].values)
                    else:
                        Ref_Data[cols[i]] = df[cols[i]]
            
            Ref_Data = Ref_Data[sorted(cols)]
        
            if inconsistencies != "":
                print("Inconsistent indices in files "+inconsistencies)

        Ref_Data.index.name = "Energy (cm$^{-1}$)"
        Ref_Data = self.set_units(data_ch = Ref_Data, is_ref = True)
        return Ref_Data
    
    def Normalise(self, Loaded_Data, Ref_Data):
        """Normalises the dataframe using the loaded measurement and zero-field data

        Args:
            Loaded_Data (pandas dataframe): Measured data
            Ref_Data (pandas dataframe): Reference data

        Returns:
            pandas dataframe: Normalised spectral data
        """
        if len(Ref_Data.columns) == 1: # No interpolation is possible if there is only one zero-field dataset
            return Loaded_Data.div(Ref_Data.values, axis=0)
        else:
            Interpolated_Data = Treatment.Interpolate(Ref_Data, Ref_Data.columns, Loaded_Data.columns)
            return Loaded_Data.div(Interpolated_Data)

    def Correct(self, Loaded_Data, Ref_Data, cor):
        """Corrects the data in the given folder

        Args:
            Loaded_Data (pandas dataframe): Measured data
            Ref_Data (pandas dataframe): Reference data
            cor (bool): Check whether the data is pre-corrected

        Returns:
            pandas dataframe: Dataframe corrected for reference
        """
        cols = Loaded_Data.columns
        Ref_Data = Treatment.Interpolate(Ref_Data, Ref_Data.columns, cols)
        E_vals = Loaded_Data.index

        # Select window and create interpolated reference
        Ref_Data = Ref_Data.loc[min(E_vals):max(E_vals)]

        for i in range(0,np.size(cols)):
            if cor == False: # No need to correct data which is already corrected
                Ref_Interpol = np.interp(Loaded_Data.index, Ref_Data.index, Ref_Data[cols[i]].values) # Column-wise correction
                Loaded_Data[cols[i]] = Loaded_Data[cols[i]].div(Ref_Interpol)
        return Loaded_Data

    def Do_All(self):
        """Average data where multiple runs have been taken and generated a corrected dataframe or dictionary of dataframes

        Returns:
            dict or dataframe: Corrected spectral data, or set of spectral datafiles for each window
        """
        if self.as_folder == False:
            Loaded_Data = self.Load_Data(self.data_path)
            Ref_Data = self.Load_Refs(self.ref_path)
            if self.zero_field == True:
                Norm_df = self.Normalise(Loaded_Data, Ref_Data)
            else:
                FileList = self.file_list(self.data_path, is_ref = False)
                cor = False
                for name in FileList:
                    if ".cor" in name:
                        cor = True
                Norm_df = self.Correct(Loaded_Data, Ref_Data, cor)
            return Norm_df
        
        else:
            Norm_dict = {}
            # Load references. If the data share a path then the normalisation should uses the reference for that specific folder.
            # Otherwise, the normalisation should use one reference for the whole dataset.
            if self.sep_paths == True:
                ref_data_dict = {}
                Ref_Data = self.Load_Refs(self.folder_refs[0][0])
                for i in range(0, len(self.folder_list)):
                    ref_data_dict[i] = Ref_Data # Create copies of the reference data.
                ref_paths_list = np.tile(self.folder_refs[0][0], (len(self.folder_list),1))
            else:
                ref_paths_list = []
                ref_data_dict = {}
                for i in range(0,len(self.folder_refs)):
                    df_list = []
                    for subfolder in self.folder_refs[i]: # Take mean value of the merged dataframes for each folder in a given window
                        df = self.Load_Refs(subfolder)
                        df_list.append(df)
                    if len(df_list) == 1:
                        Ref_Data = df_list[0] # Concat is not possible for only one dataframe
                    else:
                        Ref_Data = pd.concat(df_list)
                        Ref_Data = Ref_Data.groupby(by=Ref_Data.index.name).mean()
                        Ref_Data = Ref_Data[sorted(Ref_Data.columns)]
                    ref_data_dict[i] = Ref_Data
                    ref_paths_list.append(self.folder_refs[i][0])
            
            # Load measurement data
            for i in range(0,len(self.folder_list)):
                Ref_Data = ref_data_dict[i]
                df_list = []
                for subfolder in self.folder_paths[i]:
                    df = self.Load_Data(subfolder)
                    df_list.append(df)
                if len(df_list) == 1:
                    Loaded_Data = df_list[0]
                else:
                    Loaded_Data = pd.concat(df_list)
                    Loaded_Data = Loaded_Data.groupby(by=Loaded_Data.index.name).mean()
                    Loaded_Data = Loaded_Data[sorted(Loaded_Data.columns)]

                if self.zero_field == True:
                    Norm_df = self.Normalise(Loaded_Data, Ref_Data)
                    Norm_dict[self.folder_list[i]] = Norm_df
                
                else:
                    FileList = self.file_list(subfolder, is_ref = False)
                    cor = False
                    for name in FileList:
                        if ".cor" in name:
                            cor = True
                    Norm_df = self.Correct(Loaded_Data, Ref_Data, cor)
                    Norm_dict[self.folder_list[i]] = Norm_df
            
            # Merge overlapping windows and resort
            Norm_df = Treatment.merge_windows(Norm_dict, high_to_low = self.high_to_low)
            return Norm_df

class Treatment(object):
    """Compilation of static methods for better organisation. Adapted from the previous iteration of this module made by Jan Wyzula
    """
    @staticmethod
    def merge_windows(dict, high_to_low=None):
        """Merge the energy windows and check for duplicates and ordered indexes

        Args:
            dict (dictionary): Dictionary of normalised dataframes, with keys in order of energy
            high_to_low (Boolean): Check if baselines should be adjusted high-to-low or low-to-high

        Returns:
            pandas dataframe: Dataframe of merged windows
        """
        tempdict = dict.copy()
        keys = list(tempdict.keys())
        n_keys = len(keys)

        if high_to_low == True:
            adjust = "high_to_low"
        elif high_to_low == False:
            adjust = "low_to_high"
        else:
            adjust_input = input("Adjust baselines high to low? Y/N: ")
            if adjust_input == "Y":
                adjust = "high_to_low"
            elif adjust_input == "N":
                adjust = "low_to_high"
            else:
                print("Unrecognised input. Adjusting high to low.")
                adjust = "high_to_low"
    
        for i in range(0,n_keys-1):
            if adjust == "low_to_high": # Starts from the end of the list and changes the lower baseline to match the higher one in each pair
                tempdict[keys[-i-2]] = Treatment.match_baseline(tempdict[keys[-i-2]],tempdict[keys[-i-1]],adjust="low_to_high")
            elif adjust == "high_to_low": # Starts from the front of the list and changes the higher baseline to match the lower one in each pair
                tempdict[keys[i+1]] = Treatment.match_baseline(tempdict[keys[i]],tempdict[keys[i+1]],adjust="high_to_low")
        
        df = pd.concat(list(tempdict.values()))
        # Re-sort the indexes and columns and check for duplicates
        df = df.sort_index()
        df = df.groupby(by=df.index.name).mean()
        if type(df) is not pd.Series:
            df = df[sorted(df.columns)]

        return df

    @staticmethod
    def derivative(data, axis=0, edge=1):
        """Making derivative of the measuremnt.

        Args:
            data (pd.DataFrame): 2D measurememnt matrix, columns - magnetic field, index - energy
            axis (int, optional): switch for derivation along energy or magnetic field. Defaults to 0 (energy), 1 (magnetic field).
            edge (int, optional): How to deal with edges of dataset, refer to numpy.gradient documentation. Defaults to 1.

        Returns:
            pd.DataFrame: Derivative of data
        """
        grad = np.gradient(data.values, edge, axis=axis)  # [0]
        data_der_pd = pd.DataFrame(data=grad, index=data.index, columns=data.columns)
        return data_der_pd

    @staticmethod
    def BS_correct(data, region):
        """Baseline normalisation.

        Args:
            data (pd.DataFrame): Dataframe of measuremnts with columns as the magnetic field and index as the energy
            region (tuple, list): Define region to be normalized to 1.

        Returns:
            pd.DataFrame: Corrected dataset
        """
        mean = np.mean(data.loc[region[0] : region[1]].values, axis=0) - 1
        return data - mean

    @staticmethod
    def Interpolate(data, B0, B):
        """Linear interpolation.

        Args:
            data (pd.DataFrame): Original Data
            B0 (np.array, list): Axis matching the data.
            B (np.array, list): New Axis on which dataset should be interpolated

        Returns:
            pd.DataFrame: Interpolated matrix
        """
        interpol = interp1d(B0, data.values, fill_value="extrapolate")
        return pd.DataFrame(
            data=interpol(B),
            index=data.index,
            columns=B,
        )

    @staticmethod
    def Change_units(data, units):
        """Detects which units are used and changes between cm-1 <=> meV

        Args:
            data (pd.DataFrame): Dataset

        Returns:
            pd.DataFrame: Dataset with changed units
        """
        if units == "cm-1":
            if data.index.name == "Energy (cm$^{-1}$)":
                pass
            elif data.index.name == "Energy (meV)":
                data.index *= 8.0656
            elif data.index.name == "Energy (THz)":
                data.index *= 33.35641
            data.index.name = "Energy (cm$^{-1}$)"
            return data
        
        elif units == "meV":
            if data.index.name == "Energy (meV)":
                pass
            elif data.index.name == "Energy (cm$^{-1}$)":
                data.index /= 8.0656
            elif data.index.name == "Energy (THz)":
                data.index /= (8.0656 / 33.35641 )
            data.index.name = "Energy (meV)"
            return data
        
        elif units == "THz":
            if data.index.name == "Energy (THz)":
                pass
            elif data.index.name == "Energy (cm$^{-1}$)":
                data.index /= 33.35641
            elif data.index.name == "Energy (meV)":
                data.index /= (33.35641 / 8.0656)
            data.index.name = "Energy (THz)"
            return data
        else:
            #Unrecognised unit input (must be cm-1, meV, or THz). Assuming cm-1 units.
            data.index.name = "Energy (cm$^{-1}$)"
            return data
        
    @staticmethod
    def match_baseline(df_low, df_high, adjust="low_to_high"):
        """Match the baselines of dataframes for overlapping energy windows. This should work even if "df_high" is lower in energy than "df_low".

        Args:
            df_low (dataframe): Dataframe for first window
            df_high (dataframe): Dataframe for second window
            adjust (str, optional): Checks which window should be used as the baseline. Defaults to "low_to_high"

        Raises:
            ValueError: Invalid input for "adjust"

        Returns:
            dataframe: Adjusted dataframe only
        """
        
        # Step 1: Determine the overlapping region
        overlap_min = max(df_low.index.min(), df_high.index.min())
        overlap_max = min(df_low.index.max(), df_high.index.max())

        # Extract overlapping data from both DataFrames
        overlap_low = df_low.loc[
            (df_low.index >= overlap_min) & (df_low.index <= overlap_max)
        ]
        overlap_high = df_high.loc[
            (df_high.index >= overlap_min) & (df_high.index <= overlap_max)
        ]

        # Prepare for merging: interpolate the indexes in the region of overlap
        overlap_index = overlap_low.index.union(overlap_high.index)
        new_high_index = df_high.index.union(overlap_index).sort_values()
        new_low_index = df_low.index.union(overlap_index).sort_values()
        df_high_adjusted = df_high.copy().reindex(new_high_index).interpolate()
        df_low_adjusted = df_low.copy().reindex(new_low_index).interpolate()
        weight = (overlap_index - overlap_min)/(overlap_max - overlap_min)

        # Step 2: Calculate the average baseline difference
        baseline_diff = overlap_high.mean() - overlap_low.mean()

        if adjust == "low_to_high":
            # Adjust the baseline of the lower energy window and average using weight
            if type(baseline_diff) is pd.Series:
                df_low_adjusted = df_low_adjusted.add(baseline_diff, axis=1)
            else:
                df_low_adjusted = df_low_adjusted.add(baseline_diff)
            df_low_adjusted.loc[overlap_index] = df_low_adjusted.loc[overlap_index].mul(weight,axis=0) + df_high_adjusted.loc[overlap_index].mul((1-weight),axis=0)
            return df_low_adjusted
        elif adjust == "high_to_low":
            # Adjust the baseline of the higher energy window and average using weight
            if type(baseline_diff) is pd.Series:
                df_high_adjusted = df_high_adjusted.subtract(baseline_diff, axis=1)
            else:
                df_high_adjusted = df_high_adjusted.subtract(baseline_diff)
            df_high_adjusted.loc[overlap_index] = df_low_adjusted.loc[overlap_index].mul(weight,axis=0) + df_high_adjusted.loc[overlap_index].mul((1-weight),axis=0)
            return df_high_adjusted
        else:
            raise ValueError(
                "Invalid value for 'adjust' parameter. Choose either 'low_to_high' or 'high_to_low'."
            )

    @staticmethod
    def SG_smooth(data, window, poly):
        """Savitzky–Golay filter for smoothing of 2D pd.DataFrame

        Args:
            data (pd.DataFrame): DataSet
            window (tuple, list): Size of smoothing window.
            poly (int): Polynomian order, has to be odd number

        Returns:
            pd.DataFrame: Smoothed data.
        """
        data_s = pd.DataFrame(
            data=savgol_filter(data.iloc[:, 0], window, poly),
            index=data.index,
            columns=[data.columns[0]],
        )
        for i in range(1, np.size(data.columns)):
            data_s[data.columns[i]] = pd.Series(
                savgol_filter(data.iloc[:, i], window, poly), index=data_s.index
            )
        return data_s

    @staticmethod
    def Kramers_Kronig(df, model, low_lim, high_lim, below_fe, b=None):
        """Compute Kramers-Kronig analysis for the given data

        Args:
            df (pandas dataframe): Single column of dataframe with reflectivity data. Index should be energy in Hz
            model (string): Type of model to be used for low-energy extrapolation
            low_lim (float): Energy limit to be used for low-energy extraploation
            high_lim (float): Energy limit to be used for high-energy extrapolation
            below_fe (Boolean): Check whether the upper limit of the energy range is below the free-electron (plasma) frequency
            b (float, optional): Additional parameter for the insulator and power law models. Defaults to None.
    
        Returns:
            pandas dataframe: Dataframe with the energy in Hz as the index and columns for: 
            the real part of the dielectric function "er";
            the imaginary part of the dielectric function "ei";
            the magnitude of reflectivity "rf";
            the real part of the optical conductivity "sr";
            and the phase of the reflectivity "phase"
        """
        pi = scipy.constants.pi
        e0 = scipy.constants.epsilon_0
         
        def func_low(w, a):
            """Low-energy limit equation

            Args:
                w (float): Independent variable
                a (float): Model parameter

            Returns:
                float: Dependent variable
            """
            if model == "Hagen-Rubens":
                return 1 - a*np.sqrt(w)
            elif model == "Insulator":
                return b + a*w**2
            elif model == "Power law":
                return 1 - a*w**(-1/b)
            elif model == "Metal":
                return 1 - a/np.sqrt(w)
            elif model == "Marginal Fermi liquid":
                return 1 - a/w
            elif model == "Gorter-Casimir two-fluid model":
                return 1 - a/w**2
            elif model == "Superconducting":
                return 1 - a/w**4
        
        def func_high(w, a):
            """High-energy limit equation

            Args:
                w (float): Independent variable
                a (string)): Model parameter

            Returns:
                float: Dependent variable
            """
            if below_fe == True:
                return 1 - a/w**4
            else:
                return a
        

        # Extrapolate to the low-energy limit
        w_data = df.loc[:low_lim].index
        R_data = df.loc[:low_lim].values
        popt,pcov = curve_fit(func_low, w_data, R_data)
        a_low = popt[0]

        # Extrapolate to the high-energy limit
        w_data = df.loc[high_lim:].index
        R_data = df.loc[high_lim:].values
        popt,pcov = curve_fit(func_high, w_data, R_data)
        a_high = popt[0]

        # Calculate phase using Kramers-Kronig integral
        theta = []
        w_vals = list(df.index)
        w_min = w_vals[0]
        w_max = w_vals[-1]
        for i in range(0, len(w_vals)):
            w0 = w_vals[i]
            # Integrate over low energies
            intg_low = lambda x: np.log(func_low(x, a_low))/(x**2-w0**2)
            theta_low = -w0/pi * quad(intg_low, 0.0, w_min)[0] # There are usually singularities here at w0 = 0 and w0 = low_lim
            # Integrate over high energies
            intg_high = lambda x: np.log(func_high(x, a_high))/(x**2-w0**2)
            theta_high = -w0/pi * quad(intg_high, w_max, np.inf)[0] # There is usually a singularity here at w0 = high_lim

            # Integrate between the two limits
            h = df.loc[w_vals]
            for j in range(0, len(w_vals)):
                w = w_vals[j]
                if w == w0: # Dealing with singularities: use l'Hopital's rule
                    if w == w_min:
                        slope=-(df.loc[w_vals[j+1]]-df.loc[w])/(w_vals[j+1]-w)
                        h.loc[w]=slope/(2.0*w*df.loc[w])
                    elif w == w_max:
                        slope=-(df.loc[w]-df.loc[w_vals[j-1]])/(w-w_vals[j-1])
                        h.loc[w]=slope/(2.0*w*df.loc[w])
                    else:
                        slope=-(df.loc[w_vals[j+1]]-df.loc[w_vals[j-1]])/(w_vals[j+1]-w_vals[j-1])
                        h.loc[w]=slope/(2.0*w*df.loc[w])
                else:
                    h.loc[w] = np.log(df.loc[w])/(w**2-w0**2)
            theta_mid = -w0/pi*trapz(h.values,h.index)
            theta.append(theta_low + theta_mid + theta_high) # Array with the same length as the original dataframe
    
        # Calculate real and imaginary parts of refractive index, dielectric constant, and conductivity
        R = np.absolute(list(df.values)) # Take absolute value in case some are negative - later calculations involve taking the square root
        n = (1-R)/(1+2*np.cos(theta)*np.sqrt(R)+R)
        k = (-2*np.sqrt(R)*np.sin(theta))/(1+2*np.cos(theta)*np.sqrt(R)+R)
        er = n**2 - k**2
        ei = 2*n*k
        sr = w_vals*ei 
        sr *= e0

        # Save er, ei, R, sr, and theta to a dataframe
        data = {"er": er, "ei": ei, "R": R, "sr": sr, "phase": theta}
        proc_df = pd.DataFrame(data=data, index=w_vals)
        return proc_df
    