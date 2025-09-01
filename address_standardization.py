# Install: pip install usaddress tqdm
import usaddress
import pandas as pd
import re
import os
from tqdm import tqdm

from pathlib import Path
def read_file(file_path, **kwargs):
    """
    Read CSV or Parquet file based on file extension
    
    Args:
        file_path (str): Path to the file
        **kwargs: Additional arguments to pass to pandas read functions
    
    Returns:
        pd.DataFrame: The loaded dataframe
    """
    # Get file extension
    file_extension = Path(file_path).suffix.lower()
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read based on extension
    if file_extension == '.csv':
        return pd.read_csv(file_path, **kwargs)
    elif file_extension == '.parquet':
        return pd.read_parquet(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: .csv, .parquet")
def standardize_with_usaddress_separate_unit(address_string):
    """
    Standardize address using usaddress library - separates unit info
    Returns tuple: (standardized_address, unit_info)
    """
    if pd.isna(address_string) or address_string == '':
        return '', ''
    try:
        # Pre-clean: Remove periods for more consistent parsing, strip extra spaces
        cleaned_address = re.sub(r'\.', '', str(address_string)).upper().strip()
        cleaned_address = re.sub(r'\s+', ' ', cleaned_address)

        # Parse the address into components
        parsed = usaddress.tag(cleaned_address)
        components = parsed[0]  # First element is the parsed components

        # Build standardized address (without unit)
        standardized_parts = []
        unit_parts = []

        # Add house number
        if 'AddressNumber' in components:
            standardized_parts.append(components['AddressNumber'])

        # Add street direction prefix (EXPANDED)
        if 'StreetNamePreDirectional' in components:
            direction = components['StreetNamePreDirectional'].upper()
            # Map common directional abbreviations TO their full forms
            direction_map_expanded = {
                'N': 'NORTH', 'S': 'SOUTH', 'E': 'EAST', 'W': 'WEST',
                'NE': 'NORTHEAST', 'NW': 'NORTHWEST',
                'SE': 'SOUTHEAST', 'SW': 'SOUTHWEST',
                'NORTH': 'NORTH', 'SOUTH': 'SOUTH', 'EAST': 'EAST', 'WEST': 'WEST', # Map full names to themselves
                'NORTHEAST': 'NORTHEAST', 'NORTHWEST': 'NORTHWEST',
                'SOUTHEAST': 'SOUTHEAST', 'SOUTHWEST': 'SOUTHWEST'
            }
            expanded_direction = direction_map_expanded.get(direction, direction)
            standardized_parts.append(expanded_direction)

        # Add street name
        if 'StreetName' in components:
            # Apply expansion to potential abbreviations within the street name part itself if needed later,
            # but usaddress usually separates type. For now, just add the name part.
            standardized_parts.append(components['StreetName'].upper())

        # Add street type (EXPANDED)
        if 'StreetNamePostType' in components:
            street_type = components['StreetNamePostType'].upper()
            # Expand common street type abbreviations
            street_type_expand = {
                'ST': 'STREET', 'STR': 'STREET', 'STREEET': 'STREET', 'STREET': 'STREET',
                'AVE': 'AVENUE', 'AV': 'AVENUE', 'AVENU': 'AVENUE', 'AVENUE': 'AVENUE',
                'RD': 'ROAD', 'ROAD': 'ROAD', 'ROADS': 'ROADS', # Keep ROADS plural as is if parsed
                'DR': 'DRIVE', 'DRIV': 'DRIVE', 'DRV': 'DRIVE', 'DRIVE': 'DRIVE',
                'LN': 'LANE', 'LANE': 'LANE', 'LANES': 'LANES', # Keep LANES plural as is
                'CT': 'COURT', 'CRT': 'COURT', 'COURT': 'COURT',
                'PL': 'PLACE', 'PLC': 'PLACE', 'PLACE': 'PLACE',
                'BLVD': 'BOULEVARD', 'BOUL': 'BOULEVARD', 'BL': 'BOULEVARD', 'BOULEVARD': 'BOULEVARD',
                'PKWY': 'PARKWAY', 'PKY': 'PARKWAY', 'PARKWY': 'PARKWAY', 'PARKWAY': 'PARKWAY',
                'CIR': 'CIRCLE', 'CIRC': 'CIRCLE', 'CRCLE': 'CIRCLE', 'CIRCLE': 'CIRCLE',
                'WAY': 'WAY', 'WY': 'WAY', 'WAY': 'WAY', # WAY is often not abbreviated, but WY might be
                'TRL': 'TRAIL', 'TR': 'TRAIL', 'TRAIL': 'TRAIL',
                'HWY': 'HIGHWAY', 'HW': 'HIGHWAY', 'HIWY': 'HIGHWAY', 'HIGHWAY': 'HIGHWAY',
                'TER': 'TERRACE', 'TERR': 'TERRACE', 'TERRACE': 'TERRACE',
                'ALY': 'ALLEY', 'AL': 'ALLEY', 'ALLEE': 'ALLEY', 'ALLEY': 'ALLEY'
            }
            expanded_type = street_type_expand.get(street_type, street_type)
            standardized_parts.append(expanded_type)

        # Add street direction suffix (EXPANDED)
        if 'StreetNamePostDirectional' in components:
            direction = components['StreetNamePostDirectional'].upper()
            # Expand common directional abbreviations
            direction_map_expanded = {
                'N': 'NORTH', 'S': 'SOUTH', 'E': 'EAST', 'W': 'WEST',
                'NE': 'NORTHEAST', 'NW': 'NORTHWEST',
                'SE': 'SOUTHEAST', 'SW': 'SOUTHWEST',
                'NORTH': 'NORTH', 'SOUTH': 'SOUTH', 'EAST': 'EAST', 'WEST': 'WEST',
                'NORTHEAST': 'NORTHEAST', 'NORTHWEST': 'NORTHWEST',
                'SOUTHEAST': 'SOUTHEAST', 'SOUTHWEST': 'SOUTHWEST'
            }
            expanded_direction = direction_map_expanded.get(direction, direction)
            standardized_parts.append(expanded_direction)

        # Collect unit information separately (EXPANDED)
        if 'OccupancyType' in components:
            unit_type_raw = components['OccupancyType'].upper()
            # Normalize and then expand unit type abbreviations
            unit_type_normalize = {
                'APT': 'APARTMENT', 'APARTMENT': 'APARTMENT', 'APTMENT': 'APARTMENT',
                'STE': 'SUITE', 'SUITE': 'SUITE', 'SUIT': 'SUITE',
                'UNIT': 'UNIT', 'UNT': 'UNIT',
                'FL': 'FLOOR', 'FLR': 'FLOOR', 'FLOOR': 'FLOOR',
                'BLDG': 'BUILDING', 'BLD': 'BUILDING', 'BUILDING': 'BUILDING'
            }
            normalized_unit = unit_type_normalize.get(unit_type_raw, unit_type_raw)

            unit_type_expand = {
                'APARTMENT': 'APARTMENT', 'SUITE': 'SUITE',
                'UNIT': 'UNIT', 'FLOOR': 'FLOOR', 'BUILDING': 'BUILDING'
            }
            expanded_unit = unit_type_expand.get(normalized_unit, normalized_unit)
            unit_parts.append(expanded_unit)

            if 'OccupancyIdentifier' in components:
                unit_parts.append(components['OccupancyIdentifier'])

        standardized_address = ' '.join(standardized_parts)
        unit_info = ' '.join(unit_parts) if unit_parts else ''
        return standardized_address, unit_info

    except Exception as e:
        # If parsing fails, fall back to basic cleaning
        # print(f"usaddress parsing failed for '{address_string}': {e}") # Optional debug print
        return basic_standardization_separate_unit(address_string)

def standardize_state(state):
    """
    Standardize state names/abbreviations to uppercase full state names. 
    """
    if pd.isna(state) or state == '':
        return ''
    state = str(state).strip().upper()
    # Map common state abbreviations and names to full names
    state_full_names = {
        # Standard two-letter abbreviations
        'AL': 'ALABAMA', 'AK': 'ALASKA', 'AZ': 'ARIZONA', 'AR': 'ARKANSAS', 'CA': 'CALIFORNIA',
        'CO': 'COLORADO', 'CT': 'CONNECTICUT', 'DE': 'DELAWARE', 'FL': 'FLORIDA', 'GA': 'GEORGIA',
        'HI': 'HAWAII', 'ID': 'IDAHO', 'IL': 'ILLINOIS', 'IN': 'INDIANA', 'IA': 'IOWA',
        'KS': 'KANSAS', 'KY': 'KENTUCKY', 'LA': 'LOUISIANA', 'ME': 'MAINE', 'MD': 'MARYLAND',
        'MA': 'MASSACHUSETTS', 'MI': 'MICHIGAN', 'MN': 'MINNESOTA', 'MS': 'MISSISSIPPI',
        'MO': 'MISSOURI', 'MT': 'MONTANA', 'NE': 'NEBRASKA', 'NV': 'NEVADA', 'NH': 'NEW HAMPSHIRE',
        'NJ': 'NEW JERSEY', 'NM': 'NEW MEXICO', 'NY': 'NEW YORK', 'NC': 'NORTH CAROLINA',
        'ND': 'NORTH DAKOTA', 'OH': 'OHIO', 'OK': 'OKLAHOMA', 'OR': 'OREGON', 'PA': 'PENNSYLVANIA',
        'RI': 'RHODE ISLAND', 'SC': 'SOUTH CAROLINA', 'SD': 'SOUTH DAKOTA', 'TN': 'TENNESSEE',
        'TX': 'TEXAS', 'UT': 'UTAH', 'VT': 'VERMONT', 'VA': 'VIRGINIA', 'WA': 'WASHINGTON',
        'WV': 'WEST VIRGINIA', 'WI': 'WISCONSIN', 'WY': 'WYOMING',
        # Ensure full names map to themselves
        'ALABAMA': 'ALABAMA', 'ALASKA': 'ALASKA', 'ARIZONA': 'ARIZONA', 'ARKANSAS': 'ARKANSAS',
        'CALIFORNIA': 'CALIFORNIA', 'COLORADO': 'COLORADO', 'CONNECTICUT': 'CONNECTICUT',
        'DELAWARE': 'DELAWARE', 'FLORIDA': 'FLORIDA', 'GEORGIA': 'GEORGIA', 'HAWAII': 'HAWAII',
        'IDAHO': 'IDAHO', 'ILLINOIS': 'ILLINOIS', 'INDIANA': 'INDIANA', 'IOWA': 'IOWA',
        'KANSAS': 'KANSAS', 'KENTUCKY': 'KENTUCKY', 'LOUISIANA': 'LOUISIANA', 'MAINE': 'MAINE',
        'MARYLAND': 'MARYLAND', 'MASSACHUSETTS': 'MASSACHUSETTS', 'MICHIGAN': 'MICHIGAN',
        'MINNESOTA': 'MINNESOTA', 'MISSISSIPPI': 'MISSISSIPPI', 'MISSOURI': 'MISSOURI',
        'MONTANA': 'MONTANA', 'NEBRASKA': 'NEBRASKA', 'NEVADA': 'NEVADA', 'NEW HAMPSHIRE': 'NEW HAMPSHIRE',
        'NEW JERSEY': 'NEW JERSEY', 'NEW MEXICO': 'NEW MEXICO', 'NEW YORK': 'NEW YORK',
        'NORTH CAROLINA': 'NORTH CAROLINA', 'NORTH DAKOTA': 'NORTH DAKOTA', 'OHIO': 'OHIO',
        'OKLAHOMA': 'OKLAHOMA', 'OREGON': 'OREGON', 'PENNSYLVANIA': 'PENNSYLVANIA',
        'RHODE ISLAND': 'RHODE ISLAND', 'SOUTH CAROLINA': 'SOUTH CAROLINA', 'SOUTH DAKOTA': 'SOUTH DAKOTA',
        'TENNESSEE': 'TENNESSEE', 'TEXAS': 'TEXAS', 'UTAH': 'UTAH', 'VERMONT': 'VERMONT',
        'VIRGINIA': 'VIRGINIA', 'WASHINGTON': 'WASHINGTON', 'WEST VIRGINIA': 'WEST VIRGINIA',
        'WISCONSIN': 'WISCONSIN', 'WYOMING': 'WYOMING'
    }
    return state_full_names.get(state, state) # Return original if not found

def basic_standardization_separate_unit(address):
    """Fallback standardization if usaddress parsing fails - expands abbreviations and separates unit info"""
    if pd.isna(address):
        return "", ""
    # Pre-clean: Remove periods for more consistent regex matching, convert to upper, strip
    address = re.sub(r'\.', '', str(address)).upper().strip()
    address = re.sub(r'\s+', ' ', address)

    # Extract unit information (EXPANDED)
    # Define patterns to match various unit formats and capture the identifier
    # Order matters: match more specific or longer patterns first if needed, though these are distinct
    unit_patterns_expand = [
        (r'\b(APT|APARTMENT|APTMENT)\s*([A-Z0-9\-]+)', 'APARTMENT'),
        (r'\b(STE|SUITE|SUIT)\s*([A-Z0-9\-]+)', 'SUITE'),
        (r'\b(UNIT|UNT)\s*([A-Z0-9\-]+)', 'UNIT'),
        (r'\b(FL|FLR|FLOOR)\s*([A-Z0-9\-]+)', 'FLOOR'),
        (r'\b(BLDG|BLD|BUILDING)\s*([A-Z0-9\-]+)', 'BUILDING'),
    ]
    unit_info = ""
    for pattern, full_type in unit_patterns_expand:
        match = re.search(pattern, address)
        if match:
            # Use the expanded full type and the captured identifier
            unit_number = match.group(2)
            unit_info = f"{full_type} {unit_number}"
            address = re.sub(pattern, '', address).strip()
            break # Stop after finding the first match

    # Basic replacements to expand common abbreviations
    replacements_expand = {
        r'\bST\b': 'STREET', r'\bSTR\b': 'STREET', r'\bSTREEET\b': 'STREET',
        r'\bAVE\b': 'AVENUE', r'\bAV\b': 'AVENUE', r'\bAVENU\b': 'AVENUE',
        r'\bRD\b': 'ROAD', # Add potential variants if needed
        r'\bDR\b': 'DRIVE', r'\bDRIV\b': 'DRIVE', r'\bDRV\b': 'DRIVE',
        r'\bLN\b': 'LANE', # Add potential variants if needed
        r'\bCT\b': 'COURT', r'\bCRT\b': 'COURT',
        r'\bPL\b': 'PLACE', r'\bPLC\b': 'PLACE',
        r'\bBLVD\b': 'BOULEVARD', r'\bBOUL\b': 'BOULEVARD', r'\bBL\b': 'BOULEVARD', # Assume BL is Boulevard
        r'\bPKWY\b': 'PARKWAY', r'\bPKY\b': 'PARKWAY', r'\bPARKWY\b': 'PARKWAY',
        r'\bCIR\b': 'CIRCLE', r'\bCIRC\b': 'CIRCLE', r'\bCRCLE\b': 'CIRCLE',
        r'\bTRL\b': 'TRAIL', r'\bTR\b': 'TRAIL',
        r'\bHWY\b': 'HIGHWAY', r'\bHW\b': 'HIGHWAY', r'\bHIWY\b': 'HIGHWAY',
        r'\bTER\b': 'TERRACE', r'\bTERR\b': 'TERRACE',
        r'\bALY\b': 'ALLEY', r'\bAL\b': 'ALLEY', r'\bALLEE\b': 'ALLEY', # Assume AL is Alley
        # Directions
        r'\bN\b': 'NORTH', r'\bS\b': 'SOUTH',
        r'\bE\b': 'EAST', r'\bW\b': 'WEST', r'\bNE\b': 'NORTHEAST',
        r'\bNW\b': 'NORTHWEST', r'\bSE\b': 'SOUTHEAST', r'\bSW\b': 'SOUTHWEST'
    }
    for pattern, replacement in replacements_expand.items():
        address = re.sub(pattern, replacement, address)

    # Clean up extra spaces one final time
    address = re.sub(r'\s+', ' ', address).strip()
    return address, unit_info

def standardize_addresses(input_file, address_col='address', city_col='city',
                                 state_col='state', zip_col='zip', show_progress=True):
    """
    Standardize addresses in a CSV file and save to a new file - expands abbreviations and separates unit info with progress bar
    Parameters:
    input_file (str): Path to input CSV file
    address_col (str): Name of address column
    city_col (str): Name of city column
    state_col (str): Name of state column
    zip_col (str): Name of zip column
    show_progress (bool): Whether to show progress bar
    """
    try:
        # Read the CSV file
        #df = pd.read_csv(input_file)
        df=read_file(input_file)
        print(f"Loaded {len(df)} rows from {input_file}")
        # Check if required columns exist
        required_cols = [address_col, city_col, state_col, zip_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        # Initialize lists for results
        address_results = []
        address2_results = []
        # Set up progress bar
        if show_progress:
            print("Processing addresses...")
            pbar = tqdm(total=len(df), desc="Standardizing", unit="addresses")
        else:
            pbar = None
        # Process addresses row by row
        for idx, row in df.iterrows():
            try:
                # Process the address part
                std_addr, unit_info = standardize_with_usaddress_separate_unit(
                    row[address_col] if address_col in row else ''
                )
                address_results.append(std_addr)
                address2_results.append(unit_info if unit_info else '')
            except Exception as e:
                # Handle any row-specific errors
                # print(f"Error processing row {idx}: {e}") # Optional debug print
                address_results.append('')
                address2_results.append('')
                if show_progress:
                    pbar.set_postfix({'Error': f'Row {idx}'}, refresh=False)
            # Update progress bar
            if pbar:
                pbar.update(1)
        # Close progress bar
        if pbar:
            pbar.close()
        # Add new columns to dataframe
        # Removed .str.replace('.','') as it's less relevant for expansion and handled in cleaning
        df['standardized_address'] = address_results
        df['standardized_address2'] = address2_results
        df['standardized_city'] = df[city_col].apply(lambda x: str(x).upper().strip() if pd.notna(x) else '')
        df['standardized_state'] = df[state_col].apply(lambda x: standardize_state(x) if pd.notna(x) else '')
        df['standardized_zip'] = df[zip_col].apply(lambda x: str(x).zfill(5)[:5] if pd.notna(x) else '')
        # Generate output filename
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_standardized.csv"
        # Save to new CSV file
        df.to_csv(output_file, index=False)
        print(f"\nStandardized addresses saved to {output_file}")
        print(f"Columns created: standardized_address, standardized_address2")
        print("All original columns are preserved in the output file.")
        # Show some statistics
        non_empty_address2 = sum(1 for addr2 in address2_results if addr2)
        print(f"Addresses with unit information: {non_empty_address2}/{len(df)} ({non_empty_address2/len(df)*100:.1f}%)")
        return df, output_file
    except Exception as e:
        print(f"Error processing file: {e}")
        return None, None

# Example usage and testing
if __name__ == "__main__":
    # Test the standardization function with example addresses
    test_addresses = [
        "101 W Jim Rd",
        "200 E Main St Apt 5",
        "300 N Oak Ave",
        "400 S Elm Blvd Ste 200",
        "300 N Oak Hwy",
        "500 Main St Unit B",
        "600 1st Ave Fl 3",
        "1092 SOUTH GRAND AVE.",
        "1092 S. GRAND AVENUE E. FL. 1"
    ]
    print("Testing address standardization with abbreviation expansion and unit separation:")
    print("Original -> Standardized Address | Address2")
    print("-" * 70)
    for addr in test_addresses:
        standardized_addr, unit_info = standardize_with_usaddress_separate_unit(addr)
        unit_display = unit_info if unit_info else "(empty)"
        print(f"{addr:<35} -> {standardized_addr:<40} | {unit_display}")
    print()
    print("="*70)
    print("To standardize a csv/parquet file, use:")
    print("df, output_file = standardize_addresses('your_file.csv/parquet', 'address', 'city', 'state', 'zip')")

    # Uncomment the lines below to run the CSV processing interactively
    csv_file = input("Enter csv address file: ")

    df, output_file = standardize_addresses(
        csv_file,
        address_col='address',
        city_col='city',
        state_col='state',
        zip_col='zip'
    )

