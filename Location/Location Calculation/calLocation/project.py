import pyproj
import csv


def convert_coordinates(input_csv, output_csv):
    """
    Convert WGS84 coordinates in a CSV file to UTM coordinates and write to a new CSV file.

    Args:
        input_csv: Path to the input CSV file.
        output_csv: Path to the output CSV file.
    """

    try:
        wgs84 = pyproj.CRS("EPSG:4326")  # WGS84 coordinate system
        utm = pyproj.CRS("EPSG:32650")   # UTM coordinate system (for Zone 50)
        transformer = pyproj.Transformer.from_crs(wgs84, utm)

        with open(input_csv, 'r', encoding='utf-8') as infile, open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # Write the header (assumes the first row in the input CSV is the header)
            header = next(reader)
            writer.writerow(header + ["UTM_X", "UTM_Y"])  # Add UTM coordinate columns
            i = 1
            for row in reader:
                try:
                    lon = float(row[0])  # Longitude
                    lat = float(row[1])  # Latitude
                    x, y = transformer.transform(lat, lon)  # Note the order of lat, lon
                    print(i)
                    i += 1
                    writer.writerow(row + [x, y])  # Write the row with UTM coordinates
                except (ValueError, IndexError) as e:
                    print(f"Error: Failed to process row {row}: {e}")
                    # You can choose to skip the error row or handle the error in another way
                    # writer.writerow(row + ["Error", "Error"])  # For example, write an error marker

    except FileNotFoundError:
        print(f"Error: The input file {input_csv} does not exist.")
    except Exception as e:
        print(f"Error: An unknown error occurred: {e}")


# Example usage
input_file = r"C:\data\new_structure\xy_wgs84.csv"  # Replace with your input CSV file path
output_file = r"C:\data\new_structure\xy_utm.csv"  # Replace with your output CSV file path
convert_coordinates(input_file, output_file)
print("ok")
