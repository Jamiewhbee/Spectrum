import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import scipy

class Data_macro:
    # Gets inputs, set units, and reads individual files
    def __init__(self,data_path=None,ref_path=None,auto=True,units="cm-1",data_head="none",ref_head="none",as_folder=False,col_names="B",zero_field=False,sep_paths=False,flag_txt=True):
        """Initiate instance variables for units, header checks, and Boolean checks
        """
        # Basic parameters
        self.zero_field = zero_field
        self.col_names = col_names
        self.flag_txt = flag_txt

        # Advanced parameters (not required for auto method)
        if auto == True:
            self.as_folder = False
            if ref_path == None:
                self.sep_paths = False
            else:
                self.sep_paths = True
        else:
            self.as_folder = as_folder
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
        
        # Get path(s), folder list(s), file list(s), and column names, in that order.
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
        for filename in list:
            if self.col_names == "B":
                col_vals.append(float(re.sub("p",".",filename[filename.rfind("a")+1:filename.rfind("T")])))
            elif self.col_names == "T":
                listints = re.findall(r'\d+',filename)
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
            path (str): Path to main folder
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
        """Get path to the measurement data

        Returns:
            str: Path to folder containing measurement data
        """
        data_path = input("Enter the path to the measurement data (or drag and drop): ")
        return data_path
    
    def input_ref(self):
        """Get path to the reference data

        Returns:
            str: Path to folder containing reference data
        """
        ref_path = input("Enter the path to the reference data (or drag and drop): ")
        return ref_path
    
    def input_units(self):
        """Get the units for the index. Not called by default unless "units" is set to None when the macro is called

        Returns:
            str: Either "cm-1" or "meV" or "THz"
        """
        units = input("Enter the units to be used for the index (cm-1, meV, or THz): ")
        if units == "cm-1" or units == "meV" or units == "THz":
            return units
        else:
            print("Invalid unit input (must be cm-1, meV, or THz). Using cm-1 units.")
            return "cm-1"
    
    def input_data_head(self):
        """Get method for checking the measurement file headers. See read_csv below.

        Returns:
            str: Either "all" or "none" or "check"
        """
        data_head = input("Enter method for handling measurement file headers (all/none/check): ")
        if data_head == "all" or data_head == "none" or data_head == "check":
            return data_head
        else:
            print("Invalid input: Please select all, none, or check. Using check as default.")
            return "check"
        
    def input_ref_head(self):
        """Get method for checking the reference file headers. See read_csv below.

        Returns:
            str: Either "all" or "none" or "check"
        """
        ref_head = input("Enter method for handling reference file headers (all/none/check): ")
        if ref_head == "all" or ref_head == "none" or ref_head == "check":
            return ref_head
        else:
            print("Invalid input: Please select all, none, or check. Using check as default.")
            return "check"

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
        """Read individual files, using data_head and ref_head to determine how to handle the first row

        Args:
            path (str): Path to individual file
            col (float): Column name
            is_ref (bool): Check if file is measurement or reference data

        Returns:
            pd dataframe: Loaded data
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

    def load_data(self, path):
        """Extracts measurement data from the given folder. Uses energy as the index and temperature or field as the columns

        Args:
            path (str): Path to data folder

        Returns:
            pd dataframe: Measurement data
        """
        filelist = self.file_list(path, is_ref = False)
        cols = self.get_cols(filelist)
        inconsistencies = ""

        for i in range(0,np.size(filelist)):
            full_path = os.path.join(path, filelist[i])

            if i == 0:
                load_df = self.read_csv(full_path, col=cols[i], is_ref=False)
            else:
                df = self.read_csv(full_path, col=cols[i], is_ref=False)
                if len(load_df.index) != len(df.index):
                    inconsistencies = inconsistencies + filelist[i] + ", "
                    load_df[cols[i]] = np.interp(load_df.index, df.index, df[cols[i]].values)
                elif np.allclose(load_df.index,df.index) == False:
                    inconsistencies = inconsistencies + filelist[i] + ", "
                    load_df[cols[i]] = np.interp(load_df.index, df.index, df[cols[i]].values)
                else: 
                    load_df[cols[i]] = df[cols[i]]
        
        if inconsistencies != "":
            print("Inconsistent indices in files "+inconsistencies)
        
        load_df.index.name = "Energy (cm$^{-1}$)"
        load_df = Treatment.change_units(load_df, self.units)
        load_df = load_df[sorted(cols)]
        return load_df

    def load_refs(self, path):
        """Extracts reference data from the given folder. Uses energy as the index and temperature or field as the columns

        Args:
            path (str): Path to reference folder

        Returns:
            pd dataframe: Reference data
        """
        reflist = self.file_list(path, is_ref = True)
        cols = self.get_cols(reflist)
        last_col = sorted(cols)[-1]

        if self.zero_field == True:
            full_path = os.path.join(path,reflist[0])
            ref_df = self.read_csv(full_path, col=0, is_ref=True)

            if len(reflist) >= 2: # No interpolation is possible if there is only one zero-field dataset
                full_path = os.path.join(path,reflist[1])
                df = self.read_csv(full_path, col=last_col, is_ref=True)
                ref_df[last_col] = df[last_col]

        else:
            inconsistencies = ""
            for i in range(0,np.size(reflist)):
                full_path = os.path.join(path, reflist[i])

                if i == 0:
                    ref_df = self.read_csv(full_path, col=cols[i], is_ref=True)
                else:
                    df = self.read_csv(full_path, col=cols[i], is_ref=True)
                    if len(ref_df.index) != len(df.index):
                        inconsistencies = inconsistencies + reflist[i] + ", "
                        ref_df[cols[i]] = np.interp(ref_df.index, df.index, df[cols[i]].values)
                    elif np.allclose(ref_df.index,df.index) == False:
                        inconsistencies = inconsistencies + reflist[i] + ", "
                        ref_df[cols[i]] = np.interp(ref_df.index, df.index, df[cols[i]].values)
                    else:
                        ref_df[cols[i]] = df[cols[i]]
            
            ref_df = ref_df[sorted(cols)]
            if inconsistencies != "":
                print("Inconsistent indices in files "+inconsistencies)

        ref_df.index.name = "Energy (cm$^{-1}$)"
        ref_df = Treatment.change_units(ref_df, self.units)
        return ref_df
    
    def normalize(self, load_df, ref_df):
        """Normalizes the measurement data by interpolating the reference data to the columns in the measurement data

        Args:
            load_df (pd dataframe): Measurement data
            ref_df (pd dataframe): Reference data

        Returns:
            pd dataframe: Normalized spectral data
        """
        # Interpolate by index
        union_index = ref_df.index.union(load_df.index).sort_values()
        ref_df = ref_df.reindex(union_index).interpolate(method='index').loc[load_df.index]

        if len(ref_df.columns) == 1: # No interpolation is possible if there is only one zero-field dataset
            return load_df.div(ref_df.values, axis=0)
        else:
            # Interpolate by columns
            interp_df = Treatment.interpolate(ref_df, ref_df.columns, load_df.columns)
            return load_df.div(interp_df)

    def auto(self):
        """Loads, processes, and returns a normalized dataframe for a single energy window

        Returns:
            pd dataframe: Dataframe of normalized spectral data
        """
        if isinstance(self.data_path, str):
            load_df = self.load_data(self.data_path)
            ref_df = self.load_refs(self.ref_path)
            if self.zero_field == True:
                norm_df = self.normalize(load_df, ref_df)
            else:
                filelist = self.file_list(self.data_path, is_ref = False)
                cor = False
                for name in filelist:
                    if ".cor" in name:
                        cor = True
                if cor == True:
                    norm_df = load_df
                else:
                    norm_df = self.normalize(load_df, ref_df)

        else:
            # Normalize each dataset separately, then average
            df_list = []
            for datapath in self.data_path:
                df = self.load_data(datapath)
                if self.sep_paths == False:
                    ref_df = self.load_refs(datapath)
                else:
                    ref_df = self.load_refs(self.ref_path)
                if self.zero_field == True:
                    df = self.normalize(df, ref_df)
                else:
                    fileist = self.file_list(datapath, is_ref = False)
                    cor = False
                    for name in filelist:
                        if ".cor" in name:
                            cor = True
                    if cor == False:
                        df = self.normalize(df, ref_df)   
                df_list.append(df)

            norm_df = pd.concat(df_list)
            norm_df = norm_df.groupby(by=norm_df.index.name).mean()
            norm_df = norm_df[sorted(norm_df.columns)]

        return norm_df

    def load_all(self, high_to_low=True, mult=False):
        """Loads data for multiple energy windows, normalizes, and merges into a single dataframe

        Args:
            high_to_low (bool): Parameter to be passed to adjust_baselines. Check whether baselines should be adjusted high-to-low or low-to-high. Defaults to True.
            mult (bool): Parameter to be passed to adjust_baselines. Check whether baselines should be adjusted by multiplying or by adding. Defaults to False.

        Returns:
            pd dataframe: Normalized, merged spectral data
        """
        if self.as_folder == False:
            load_df = self.load_data(self.data_path)
            ref_df = self.load_refs(self.ref_path)
            if self.zero_field == True:
                norm_df = self.normalize(load_df, ref_df)
            else:
                filelist = self.file_list(self.data_path, is_ref = False)
                cor = False
                for name in filelist:
                    if ".cor" in name:
                        cor = True
                if cor == True:
                    norm_df = load_df
                else:
                    norm_df = self.normalize(load_df, ref_df)
            return norm_df
        
        else:
            norm_dict = {}
            # Load references
            ref_df_dict = {}
            if self.sep_paths == True:
                ref_df = self.load_refs(self.folder_refs[0][0])
                for i in range(0, len(self.folder_list)):
                    ref_df_dict[i] = ref_df # Create copies of the reference data
                ref_paths_list = np.tile(self.folder_refs[0][0], (len(self.folder_list),1))
            else:
                ref_paths_list = []
                for i in range(0,len(self.folder_refs)):
                    df_list = []
                    for subfolder in self.folder_refs[i]: # Take mean value of the merged dataframes for each folder in a given window
                        df = self.load_refs(subfolder)
                        df_list.append(df)
                    if len(df_list) == 1:
                        ref_df = df_list[0] # Concat is not possible for only one dataframe
                    else:
                        ref_df = pd.concat(df_list)
                        ref_df = ref_df.groupby(by=ref_df.index.name).mean()
                        ref_df = ref_df[sorted(ref_df.columns)]
                    ref_df_dict[i] = ref_df
                    ref_paths_list.append(self.folder_refs[i][0])
            
            # Load measurement data
            for i in range(0,len(self.folder_list)):
                ref_df = ref_df_dict[i]
                df_list = []
                for subfolder in self.folder_paths[i]:
                    df = self.load_data(subfolder)
                    df_list.append(df)
                if len(df_list) == 1:
                    load_df = df_list[0]
                else:
                    load_df = pd.concat(df_list)
                    load_df = load_df.groupby(by=load_df.index.name).mean()
                    load_df = load_df[sorted(load_df.columns)]

                if self.zero_field == False:
                    filelist = self.file_list(subfolder, is_ref = False)
                    cor = False
                    for name in filelist:
                        if ".cor" in name:
                            cor = True
                    if cor == True:
                        norm_df = load_df
                    else:
                        norm_df = self.normalize(load_df, ref_df)
                else:
                    norm_df = self.normalize(load_df, ref_df)
                norm_dict[self.folder_list[i]] = norm_df
            
            # Merge overlapping windows and resort
            norm_df = Treatment.merge_windows(norm_dict, high_to_low = high_to_low, mult = mult)
            return norm_df

class Treatment(object):
    """Compilation of static methods for better organisation. Adapted from the previous iteration of this module made by Jan Wyzula
    """
    @staticmethod
    def change_units(data, units):
        """Detects which units are used for the index and changes to the desired units

        Args:
            data (pd dataframe): Spectral data
            units (str): Desired units for index

        Returns:
            pd.DataFrame: Dataset with changed units
        """
        if units == "cm-1":
            if data.index.name == "Energy (meV)":
                data.index *= 8.0656
            elif data.index.name == "Energy (THz)":
                data.index *= 33.35641
            data.index.name = "Energy (cm$^{-1}$)"
        elif units == "meV":
            if data.index.name == "Energy (cm$^{-1}$)":
                data.index /= 8.0656
            elif data.index.name == "Energy (THz)":
                data.index /= (8.0656 / 33.35641)
            data.index.name = "Energy (meV)" 
        elif units == "THz":
            if data.index.name == "Energy (cm$^{-1}$)":
                data.index /= 33.35641
            elif data.index.name == "Energy (meV)":
                data.index /= (33.35641 / 8.0656)
            data.index.name = "Energy (THz)"
        else:
            print("Unrecognized unit input (must be cm-1, meV, or THz).")
        return data
        
    @staticmethod
    def interpolate(data, B0, B):
        """Linear interpolation.

        Args:
            data (pd dataframe): Original Data
            B0 (array, list): Axis matching the data.
            B (array, list): New Axis on which dataset should be interpolated

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
    def show_windows(dict, col=None):
        """Plot the separate dataframes in a dictionary

        Args:
            dict (dict): Dictionary of normalized dataframes, with keys in order of energy
            col (float, optional): Column to plot. Defaults to None.
        """
        tempdict = dict.copy()
        keys = list(tempdict.keys())

        if type(tempdict[keys[0]]) is not pd.Series:
            for i in range(0,len(keys)):
                if i == 0:
                    ax = tempdict[keys[i]][col].plot(logx=True)
                else:
                    tempdict[keys[i]][col].plot(ax=ax,logx=True)
        else:
            for i in range(0,len(keys)):
                if i == 0:
                    ax = tempdict[keys[i]].plot(logx=True)
                else:
                    tempdict[keys[i]].plot(ax=ax,logx=True)
        plt.show()
    
    @staticmethod
    def merge_windows(dict, high_to_low=None, mult=None):
        """Merge the energy windows and check for duplicates and ordered indexes.

        Args:
            dict (dict): Dictionary of normalized dataframes, with keys in order of energy
            high_to_low (bool): Parameter to be passed to adjust_baselines. Check if baselines should be adjusted high-to-low or low-to-high. Defaults to None (user input)
            mult (bool): Parameter to be passed to adjust_baselines. Check if baselines should be adjusted by multiplying or by adding. Defaults to None (user input)

        Returns:
            pd dataframe: Dataframe of merged windows
        """
        tempdict = dict.copy()
        keys = list(tempdict.keys())
        n_keys = len(keys)

        if high_to_low == None:
            adjust_input = input("Adjust baselines high to low? Y/N: ")
            if adjust_input == "Y":
                high_to_low = True
            elif adjust_input == "N":
                high_to_low = False
            else:
                print("Unrecognized input. Adjusting high to low.")
                high_to_low = True
        
        if mult == None:
            mult_input = input("Adjust baselines by adding? Y/N: ")
            if mult_input == "Y":
                mult = False
            elif mult_input == "N":
                mult = True
            else:
                print("Unrecognized input. Adjusting by adding.")
                mult = False
    
        for i in range(0,n_keys-1):
            if high_to_low == True: # Starts from the end of the list and changes the lower baseline to match the higher one in each pair
                tempdict[keys[-i-2]] = Treatment.match_baseline(tempdict[keys[-i-2]],tempdict[keys[-i-1]],high_to_low,mult)
            else: # Starts from the front of the list and changes the higher baseline to match the lower one in each pair
                tempdict[keys[i+1]] = Treatment.match_baseline(tempdict[keys[i]],tempdict[keys[i+1]],high_to_low,mult)
        
        # Merge the dataframe
        for i in range(0,n_keys):
            if i == 0:
                merged_df = tempdict[keys[i]].copy()
            else:
                merged_df = Treatment.merge(merged_df,tempdict[keys[i]])

        if type(merged_df) is not pd.Series:
            merged_df = merged_df[sorted(merged_df.columns)]

        return merged_df

    @staticmethod
    def match_baseline(df_low, df_high, high_to_low,mult):
        """Match the baselines of dataframes for overlapping energy windows. This should work even if "df_high" is lower in energy than "df_low".

        Args:
            df_low (dataframe): Dataframe for first window
            df_high (dataframe): Dataframe for second window
            high_to_low (bool, optional): Checks which baseline to adjust
            mult (bool, optional): Checks whether to adjust baselines by multiplying or by adding

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

        # Step 2: Calculate the average baseline difference
        if mult == False:
            baseline_diff = overlap_high.mean() - overlap_low.mean()
        else:
            baseline_diff = overlap_high.mean() / overlap_low.mean()

        if high_to_low == True:
            # Adjust the baseline of the lower energy window and average using weight
            if mult == False:
                if type(baseline_diff) is pd.Series:
                    df_low_adjusted = df_low.add(baseline_diff, axis=1)
                else:
                    df_low_adjusted = df_low.add(baseline_diff)
            else:
                if type(baseline_diff) is pd.Series:
                    df_low_adjusted = df_low.mul(baseline_diff, axis=1)
                else:
                    df_low_adjusted = df_low.mul(baseline_diff)
            return df_low_adjusted
        else:
            # Adjust the baseline of the higher energy window and average using weight
            if mult == False:
                if type(baseline_diff) is pd.Series:
                    df_high_adjusted = df_high.subtract(baseline_diff, axis=1)
                else:
                    df_high_adjusted = df_high.subtract(baseline_diff)
            else:
                if type(baseline_diff) is pd.Series:
                    df_high_adjusted = df_high.div(baseline_diff, axis=1)
                else:
                    df_high_adjusted = df_high.div(baseline_diff)
            return df_high_adjusted
        
    @staticmethod
    def merge(df_low, df_high):
        """Merge two dataframes and interpolate over the indexes in the overlapping region

        Args:
            df_low (pd dataframe): Dataframe with lower energy index range
            df_high (pd dataframe): Dataframe with higher energy index range

        Returns:
            pd dataframe: Merged dataframe
        """
        # Step 1: Determine the overlapping indexes
        overlap_min = max(df_low.index.min(), df_high.index.min())
        overlap_max = min(df_low.index.max(), df_high.index.max())
        overlap_index = df_low.loc[overlap_min:].index.union(df_high.loc[:overlap_max].index)

        new_high_index = df_high.index.union(overlap_index).sort_values()
        new_low_index = df_low.index.union(overlap_index).sort_values()

        # Step 2: Interpolate the dataframes in both overlapping regions to overlap_index
        overlap_low = df_low.reindex(new_low_index).interpolate(method='index').loc[overlap_min:]
        overlap_high = df_high.reindex(new_high_index).interpolate(method='index').loc[:overlap_max]

        # Step 3: Merge the two dataframes using a weight factor based on the index
        weight = (overlap_index - overlap_min)/(overlap_max - overlap_min)
        merged_overlap = overlap_low.mul((1-weight),axis=0) + overlap_high.mul(weight,axis=0)
        new_df = pd.concat([df_low.loc[df_low.index < overlap_min], merged_overlap, df_high.loc[df_high.index > overlap_max]],axis=0)
        return new_df

    @staticmethod
    def derivative(data, axis=0, edge=1):
        """Taking derivative of the measuremnt.

        Args:
            data (pd dataframe): Dataframe of measurements
            axis (int, optional): Switch for derivation along energy or magnetic field. Defaults to 0 (energy), 1 (magnetic field).
            edge (int, optional): How to deal with edges of dataset, refer to numpy.gradient documentation. Defaults to 1.

        Returns:
            pd.DataFrame: Derivative of data
        """
        grad = np.gradient(data.values, edge, axis=axis)  # [0]
        data_der_pd = pd.DataFrame(data=grad, index=data.index, columns=data.columns)
        return data_der_pd

    @staticmethod
    def bs_correct(data, region):
        """Baseline normalization.

        Args:
            data (pd dataframe): Dataframe of measurments
            region (tuple, list): Define region to be normalized to 1

        Returns:
            pd dataframe: Corrected dataset
        """
        mean = np.mean(data.loc[region[0] : region[1]].values, axis=0) - 1
        return data - mean

    @staticmethod
    def sg_smooth(data, window, poly):
        """Savitzky–Golay filter for smoothing of 2D pd dataframe

        Args:
            data (pd dataframe): Dataframe of measurements
            window (int): Size of smoothing window, must be odd
            poly (int): Polynomian order, must be less than window

        Returns:
            pd dataframe: Smoothed data
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
    def kramers_kronig(df, n, model, w_free, ptail=4, b=None):
        """Compute Kramers-Kronig analysis for the given data
        Based on code by C.C. Homes

        Args:
            df (pd dataframe): Single column of dataframe with reflectivity data
            n (int): Number of points to calculate for low-energy extrapolation
            model (str): Type of model to be used for low-energy extrapolation
            w_free (float): Free-electron frequency in same units as index
            ptail (int): Exponent for interband region. Set to 4 if w_free is within the energy range.
            b (float, optional): Additional parameter for Hagen-Rubens, insulator and power law models. Defaults to None

        Raises:
            ValueError: Invalid extrapolation model
    
        Returns:
            pd dataframe: Dataframe with the energy as the index and columns for: 
            the real part of the dielectric function "er";
            the imaginary part of the dielectric function "ei";
            the magnitude of reflectivity "rf";
            the real part of the optical conductivity "sr";
            and the phase of the reflectivity "phase"
        """
        pi = scipy.constants.pi
        e0 = scipy.constants.epsilon_0

        # Initialise the low-frequency extrapolation points for the first n points below the lowest measured frequency
        w = np.asarray(df.index)
        R = np.asarray(df.values)
        w_min = w[0]

        nums = 1 + 10**np.linspace(-2,3,n) # Array of geometrically equidistant numbers from 1.01 to 1001
        for num in nums:
            w = np.insert(w,0,w_min/num)
            R = np.insert(R,0,0)

        # Use low-frequency model to calculate extrapolation for these n points
        if model == "Hagen-Rubens":
            cond = b # Conductivity
            for i in range(0,20):
                R[i] = 1.0-np.sqrt(2.0*w[i])/(15.0*cond)

        elif model == "Insulator":
            R0 = b # Reflectance at w=0
            Rave = (R[20]+R[21]+R[22])/3.0
            a = (Rave-R0)/(w[20]**2)
            for i in range(0,20):
                R[i] = R0+a*w[i]**2

        elif model == "Power law":
            apow = b # Exponent
            a = (1.0-R[20])/(w[20]**apow)
            for i in range(0,20):
                R[i] = 1.0-a*(w[i]**apow)

        elif model == "Metal":
            a = (1.0-R[20])/np.sqrt(w[20])
            for i in range(0,20):
                R[i] = 1.0-a*np.sqrt(w[i])

        elif model == "Marginal Fermi liquid":
            a = (1.0-R[20])/w[20]
            for i in range(0,20):
                R[i] = 1.0-w[i]

        elif model == "Gorter-Casimir two-fluid model":
            a = (1.0-R[20])/(w[20]**2)
            for i in range(0,20):
                R[i] = 1.0-a*w[i]**2
        
        elif model == "Superconducting":
            a = (1.0-R[20])/(w[20]**4)
            for i in range(0,20):
                R[i] = 1.0-a*w[i]**4
        
        else:
            raise ValueError("Unrecognized extrapolation model.")

        # Set parameters for high-frequency extrapolation
        if w_free <= 0.0 or w_free <= w[-1]:
            ptail = 4
        else:
            ptail = ptail # Exponent for interband region  

        # Calculate Kramers-Kronig integration using a simple trapezoidal rule
        # Evaluate the height at each frequency point
        phase = w * 0
        for i in range(0,len(w)):
            area = 0.0
            for j in range(0,len(w)):
                if i == j:
                    if i == 0:
                        slope = (R[j+1]-R[j])/(w[j+1]-w[j])
                    elif i == len(w)-1:
                        slope = (R[j]-R[j-1])/(w[j]-w[j-1])
                    else:
                        slope = (R[j+1]-R[j-1])/(w[j+1]-w[j-1])
                    h2 = slope/(2.0*w[j]*R[j])
                else:
                    h2 = (np.log(R[i])-np.log(R[j]))/(w[j]**2-w[i]**2)
        
        # Accumulate the integral
                if j == 0:
                    h1 = h2
                else:
                    area = area+(w[j]-w[j-1])*(h1+h2)/2.0
                    h1 = h2
            phase[i] = w[i]*area/pi
        
        # Calculate the contirbution to the phase from the high-frequency component
        phase = phase - (np.log(R[-1])-np.log(R))*np.log((w[-1]+w)/(w[-1]-w))/(2*pi) + w*(ptail/w[-1]+(4-ptail)/w_free)/pi + (ptail*(w/w[-1])**3) + (4-ptail)*(w/w_free)**3/(9*pi)
        phase[-1] = phase[-2]

        # Calculate the real and imaginary parts of the dielectric function and the conductivity
        den = 1.0+R-2.0*np.sqrt(R)*np.cos(phase)
        nopt = (1.0-R)/den
        kopt = 2.0*np.sqrt(R)*np.sin(phase)/den 
        er = nopt**2 - kopt**2
        ei = 2.0*nopt*kopt
        sr = w*ei/60.0
        # si = -w*(er-einf)/60.0

        # Save er, ei, R, sr, and theta to a dataframe
        data = {"er": er, "ei": ei, "R": R, "sr": sr, "phase": phase}
        proc_df = pd.DataFrame(data=data, index=w)
        return proc_df