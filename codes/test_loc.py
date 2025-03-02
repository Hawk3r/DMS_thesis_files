import requests
import folium
import webbrowser

# API URL
url = "https://ipinfo.io/json"

try:
    response = requests.get(url)
    data = response.json()

    if "bogon" in data:
        print("Unable to retrieve location (Private network).")
    else:
        # Extract latitude and longitude
        lat, lng = map(float, data.get("loc", "0,0").split(","))

        # Print location details
        print(f"IP Address: {data.get('ip', 'N/A')}")
        print(f"City: {data.get('city', 'N/A')}")
        print(f"Region: {data.get('region', 'N/A')}")
        print(f"Country: {data.get('country', 'N/A')}")
        print(f"Latitude: {lat}")
        print(f"Longitude: {lng}")

        # Create a map centered on the location
        location_map = folium.Map(location=[lat, lng], zoom_start=15)

        # Add a marker
        folium.Marker(
            [lat, lng], 
            popup=f"Location: {data.get('city', 'Unknown')}, {data.get('region', 'Unknown')}",
            tooltip="You are here",
            icon=folium.Icon(color="red")
        ).add_to(location_map)

        # Save map to an HTML file
        map_file = "current_location_map.html"
        location_map.save(map_file)

        # Open map in browser
        webbrowser.open(map_file)

except Exception as e:
    print(f"Error: {e}")
