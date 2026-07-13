from .config import RedshiftConfig

#Output to-string for convenience
def print_config_short(config: RedshiftConfig, filepath: str, output_path: str):
    out =  "Input Audio File   :   " + filepath + "\n"
    out += "Output Mp4 File    :   " + output_path + "\n"
    out += "Output Framerate   :   {}\n".format(config.samplerate)
    out += "Colormap           :   " + config.colormap + "\n"
    out += "Cores using        :   {}\n".format(config.cores)
    return out

#More exhaustive output string
def print_config_long(config: RedshiftConfig, filepath: str, output_path: str):
    out =  "Input Audio File      :   " + filepath + "\n"
    out += "Output Mp4 File       :   " + output_path + "\n"
    out += "Output Framerate      :   {}\n".format(config.samplerate)
    out += "Cores using           :   {}\n".format(config.cores)
    out += "Batch size            :   {}\n\n".format(config.batch_size)

    out += "Colormap              :   " + config.colormap + "\n"
    out += "Colormap Range        :   ({},{})\n".format(config.colormap_range[0], config.colormap_range[1])
    out += "Colormap Periods      :   {}\n".format(config.colormap_periods)
    out += "Colormap Mirror       :   " + str(config.colormap_mirror) + "\n\n"

    out += "Frequency Mapping     :   " + config.frequency_mapping + "\n"
    out += "Flip Frequency Map    :   " + str(config.frequency_flip) + "\n"
    out += "Minimum Frequency     :   {}\n".format(config.min_frequency)
    out += "Maximum Frequency     :   {}\n".format(config.max_frequency)
    out += "Minimum Volume        :   {}\n".format(config.min_Volume) + "\n"
    out += "Volume Offset         :   " + str(config.volume_offset) + "\n"
    out += "Volume Offset Max     :   {}\n".format(config.volume_offset_max)
    out += "Volume Alpha          :   " + str(config.volume_alpha) + "\n"
    out += "Volume Alpha Range    :   ({},{})\n".format(config.volume_alpha_range[0],config.volume_alpha_range[1])
    out += "Volume Alpha Mapping  :   " + config.volume_alpha_mapping + "\n\n"

    out += "Starting radius       :   {:.8f}\n".format(config.r)
    out += "Resolution            :   {:.8f}\n".format(config.dr)
    out += "Rate of Redshift      :   {:.8f}\n\n".format(config.M)

    out += "Figure Size           :   ({},{})\n".format(config.figsize[0], config.figsize[1])
    out += "Figure X limit        :   {:.4f}\n".format(config.xlim)
    out += "Figure Y limit        :   {:.4f}\n".format(config.ylim)

    out += "Print Overall Progress:   " + str(config.print_global_progress) + "\n"
    out += "Print Render Progress :   " + str(config.print_local_progress) + "\n"
    out += "Temporary Directory   :   " + config.temp_dirs + "\n"
    out += "Delete Temp. Dir.     :   " + str(config.remove_temp_dirs) + "\n"

    return out

