#!/bin/bash

# Source and destination folders
source_folder="/path/to/MIDOG/images_raw"
destination_folder="/path/to/MIDOG/images"

# Loop through each TIFF file in the source folder
for tiff_file in "$source_folder"/*.tiff; do
    # Extract filename without extension
    filename=$(basename -- "$tiff_file")
    filename_no_ext="${filename%.*}"

    # Construct destination file path
    destination_file="${destination_folder}/${filename}"

    # Check if the destination file already exists
    if [ -f "$destination_file" ]; then
        echo "$destination_file already exists. Skipping..."
    else
        # Get file size before conversion
        size_before=$(du -sh "$tiff_file" | awk '{print $1}')

        # Display status message
        echo "Processing $filename..."

        # Run vips command to process the file
        vips tiffsave "$tiff_file" "$destination_file" --tile --tile-width 256 --tile-height 256 --pyramid

        # Get file size after conversion
        size_after=$(du -sh "$destination_file" | awk '{print $1}')

        # Check if command succeeded
        if [ $? -eq 0 ]; then
            echo "Successfully processed $filename"
            echo "File size before conversion: $size_before"
            echo "File size after conversion: $size_after"
        else
            echo "Failed to process $filename"
        fi
    fi
done
