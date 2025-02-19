// -*- coding: utf-8 -*-
// Viewer
//
// @ Fabian Hörst, fabian.hoerst@uk-essen.de
// @ Lukas Heine, lukas.heine@uk-essen.de
// Institute for Artificial Intelligence in Medicine,
// University Medicine Essen

import { Map, View } from 'ol';
import "ol-ext/dist/ol-ext.css";
import { Control, FullScreen, OverviewMap, defaults as defaultControls } from 'ol/control';
import MousePosition from 'ol/control/MousePosition.js';
import ZoomSlider from 'ol/control/ZoomSlider.js';
import TileLayer from 'ol/layer/Tile';
import Projection from 'ol/proj/Projection';
import XYZ from 'ol/source/XYZ';
import TileGrid from 'ol/tilegrid/TileGrid';
import './style.css';
import Swal from 'sweetalert2'
import VectorLayer from 'ol/layer/Vector';
import VectorSource from 'ol/source/Vector';
import { Polygon } from 'ol/geom';
import { Point } from 'ol/geom';
import { Circle as CircleStyle, Fill, Stroke, Style } from 'ol/style';

import Feature from 'ol/Feature';
import { Cluster } from 'ol/source';
import GUI from 'lil-gui';

// global variables
let colorFolder;

// post method to get the image
async function registerSlide(slideId, slidePath) {
  console.log(`Registering slide with ID: ${slideId} and image path: ${slidePath}`);
  try {
    const registerData = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        "image_id": slideId,
        "image_path": slidePath
      })
    };
    const response = await fetch("http://localhost:3306/register", registerData);
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const data = await response.json();
    console.log('POST request successful');
    console.log(data); // Log the response data
    return data;
  } catch (error) {
    console.error('There was a problem with the POST request:', error);
    throw error;
  }
}

// get method to get the slide info
async function fetchSlideInfo(slideId) {
  try {
    const response = await fetch(`http://localhost:3306/info/${slideId}`);
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const data = await response.json();
    console.log('GET request successful');
    console.log(data); // Log the response data
    return data;
  } catch (error) {
    console.error('There was a problem with the GET request:', error);
    throw error;
  }
}


// Function to check if cell detections are available
async function checkCellDetectionsExist() {
  try {
    // Perform an HTTP GET request to the server
    const response = await fetch("http://localhost:3306/detection_exists");

    // Check if the response is successful
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    // Parse the response as JSON
    let status = await response.json();
    status = status.exists;

    return status;
  } catch (error) {
    // Handle any errors that occur during the fetch
    console.error("Cannot connect to endpoint: ", error);
  }
}


// Function to check if cell contours are available
async function checkCellContoursExist() {
  try {
    // Perform an HTTP GET request to the server
    const response = await fetch("http://localhost:3306/contour_exists");

    // Check if the response is successful
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    // Parse the response as JSON
    let status = await response.json();
    status = status.exists;

    return status;
  } catch (error) {
    // Handle any errors that occur during the fetch
    console.error("Cannot connect to endpoint: ", error);
  }
}


// Function to load cell detection geojson from endpoint
async function loadCellDetections() {
  try {
    // Perform an HTTP GET request to the server
    const response = await fetch("http://localhost:3306/cell_detections");

    // Check if the response is successful
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    // Parse the response as JSON
    let geojson = await response.json();
    geojson = geojson.cell_detections;
    // Log the GeoJSON data (or handle it as needed)
    console.log(geojson);

    // Return the GeoJSON data
    return geojson;
  } catch (error) {
    // Handle any errors that occur during the fetch
    console.error("Failed to load cell detections:", error);
  }
}


// Function to load cell detection geojson from endpoint
async function loadCellContours() {
  try {
    // Perform an HTTP GET request to the server
    const response = await fetch("http://localhost:3306/cell_contours");

    // Check if the response is successful
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    // Parse the response as JSON
    let geojson = await response.json();
    geojson = geojson.cell_contours;
    // Log the GeoJSON data (or handle it as needed)
    console.log(geojson);

    // Return the GeoJSON data
    return geojson;
  } catch (error) {
    // Handle any errors that occur during the fetch
    console.error("Failed to load cell detections:", error);
  }
}

// Function to transform the cell detections into a list of points with properties
function getPointsFromMultiPointGeoJSON(slideMetadata, geojson) {
  const cellDetectionPointObjects = [];

  for (let cellDetection of geojson) {
    // first transform coordinates
    let transformedCoordinates = cellDetection.geometry.coordinates.map(point => {
        return [point[0], slideMetadata.extent[3] - point[1]];
    });

    // create a list of points from the transformed coordinates
    const points = [];
    transformedCoordinates.forEach(coord => {
      points.push(
        new Feature({
          geometry: new Point(coord)
        })
      )
    });
    cellDetectionPointObjects.push(
      {
        "points": points,
        "color": [...cellDetection.properties.classification.color, 0.65],
        "name": cellDetection.properties.classification.name
      }
    );
  }
  return cellDetectionPointObjects;
}

// Function to transform the cell contours into a list of polygons with properties
function getPolygonsFromMultiPolygonGeoJSON(slideMetadata, geojson) {
  const cellDetectionPolygonObjects = [];

  for (let cellDetection of geojson) {
    const polygons = [];
    for (let polygon of cellDetection.geometry.coordinates) {
      // problem: polygons are nested
      let transformedCoordinates = polygon[0].map(point => {
        return [point[0], slideMetadata.extent[3] - point[1]];
      });
      // create a list of polygons from the transformed coordinates
      polygons.push(
        new Feature({
          geometry: new Polygon([transformedCoordinates])
        })
      );
    };
    cellDetectionPolygonObjects.push(
      {
        "polygons": polygons,
        "color": [...cellDetection.properties.classification.color, 0.65],
        "name": cellDetection.properties.classification.name
      }
    );
  }
  return cellDetectionPolygonObjects;
}

// function to generate cell detection layers (with clustering on low zoom levels)
function generateCellDetectionLayers(cellDetectionObjects, map) {
  const cellDetectionLayers = [];

  for (let cellDetectionObject of cellDetectionObjects) {
    const clusterSource = new Cluster({
      distance: 10, // Initial distance
      source: new VectorSource({
        features: cellDetectionObject.points
      })
    });

    const styleCache = {};
    const clusterLayer = new VectorLayer({
      source: clusterSource,
      style: function (feature, resolution) {
        const size = feature.get('features').length;
        let style = styleCache[size];
        if (!style) {
          const radius = calculateRadiusForCluster(size);
          style = new Style({
            image: new CircleStyle({
              radius: radius,
              fill: new Fill({ color: cellDetectionObject.color }),
              stroke: new Stroke({ color: 'white', width: 1 })
            }),
          });
          styleCache[size] = style;
        }
        return style;
      },
      minResolution: 0.5,
      maxResolution: 40,
    });
    clusterLayer.setVisible(false);
    cellDetectionLayers.push(clusterLayer);

    // Add an event listener to change cluster size dynamically
    map.on('moveend', () => {
      const zoom = map.getView().getZoom();
      console.log(zoom)
      let distance;
      if (zoom < 12.5) {
        distance = 10;
      } else if (zoom < 13.5) {
        distance = 8;
      } else if (zoom < 15) {
        distance = 6;
      } else {
        distance = 5;
      }
      clusterSource.setDistance(distance);
    });
  }

  return cellDetectionLayers;
}

// function to generate cell contours layers
function generateCellContoursLayers(cellContoursObjects, map) {
  const cellContoursLayers = [];

  for (let cellContoursObject of cellContoursObjects) {
    const cellContoursLayer = new VectorLayer({
      source: new VectorSource({
        features: cellContoursObject.polygons
      }),
      style: new Style({
        fill: new Fill({
          color: cellContoursObject.color
        }),
        stroke: new Stroke({
          color: 'black',
          width: 1
        })
      }),
      minResolution: 0.5,
      maxResolution: 8,
    });
    cellContoursLayer.setVisible(false);
    cellContoursLayers.push(cellContoursLayer);
  }
  return cellContoursLayers
}

// Function to calculate the radius of the cluster based on the number of points
function calculateRadiusForCluster(size) {
  // Customize this function to adjust the radius based on cluster size
  const baseRadius = 3.5;
  const extraRadius = Math.log(size); // Simple example: logarithmic scaling
  return baseRadius + extraRadius;
}


// modify gui with annotations for cells
function addAnnotationFolders(gui, cellDetectionObjects) {
  colorFolder = gui.addFolder('Cell Mapping'); // Create the folder

  cellDetectionObjects.forEach(cellDetectionObject => {
    const color = `rgb(${cellDetectionObject.color[0]}, ${cellDetectionObject.color[1]}, ${cellDetectionObject.color[2]})`;

    const colorDisplay = document.createElement('div');
    colorDisplay.style.backgroundColor = color;
    colorDisplay.style.width = '20px';
    colorDisplay.style.height = '20px';
    colorDisplay.style.display = 'inline-block';
    colorDisplay.style.marginRight = '5px';

    // Create a label for the color
    const label = document.createElement('span');
    label.textContent = cellDetectionObject.name;

    // Create a container for the color display and label
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.marginBottom = '5px'; // Add some margin for spacing
    container.appendChild(colorDisplay);
    container.appendChild(label);

    // Add the custom DOM element to the folder
    const folderDom = colorFolder.__ul || colorFolder.domElement.querySelector('.children');
    if (folderDom) {
      folderDom.appendChild(container);
    }
  });

  colorFolder.open(); // Optionally open the folder by default
}


// Animations

// Function to show loading message
function showLoadingMessageSlide() {
  const mapElement = document.getElementById('map');
  mapElement.innerHTML = '<div class="loading-message"><div class="loading-spinner"></div><p>Loading Slide</p></div>';
}

// Function to show loading message
function showUploadMessage() {
  const mapElement = document.getElementById('map');
  mapElement.innerHTML = '<div class="loading-message"><div class="loading-spinner"></div><p>Uploading</p></div>';
}

// Function to show loading message when no slide is loaded
function showBackgroundMessage() {
  const mapElement = document.getElementById('map');
  mapElement.innerHTML = `
    <div class="loading-image">
      <img src="/assets/logo_background.png" alt="Loading Image" />
      <div class="dots-container">
        <div class="dots-1"></div>
      </div>
      <p>Waiting for slide</p>
    </div>
  `;
}

// Function to show loading message
function showLoadingMessageCellViT() {
  const mapElement = document.getElementById('map');
  mapElement.innerHTML = '<div class="loading-message"><div class="loading-spinner"></div><p>Loading CellViT</p></div>';
}


// Function to remove loading message
function removeLoadingMessage() {
  const mapElement = document.getElementById('map');
  mapElement.innerHTML = ''; // Remove loading message
}

// Function to initialize the WSI viewer
async function initializeWSIViewer(slideId, slidePath) {
  try {
    console.log(slideId, slidePath);
    // Wait for the slide metadata, slide and annotations (converted) to be fetched
    showLoadingMessageSlide();
    const registerStatus = await registerSlide(slideId, slidePath);
    const slideMetadata = await fetchSlideInfo(slideId);
    console.log(slideMetadata);
    // Set WSI name
    const wsiNameDiv = document.getElementById('wsi-name');
    wsiNameDiv.innerText = slideMetadata.slide_name; // Assuming wsiName is a property in slideMetadata



    const cellDetectionsExist = await checkCellDetectionsExist();
    const cellContoursExist = await checkCellContoursExist();
    let cellDetectionObjects;
    let cellContoursObjects;

    if (cellContoursExist || cellDetectionsExist) {
      console.log('CellViT results available');
      showLoadingMessageCellViT();
      if (cellDetectionsExist) {
        console.log('Cell detections available');
        const cellDetectionsGeoJson = await loadCellDetections();
        cellDetectionObjects = getPointsFromMultiPointGeoJSON(slideMetadata, cellDetectionsGeoJson);
        console.log('Loaded cell detections');
      }
      if (cellContoursExist) {
        console.log('Cell contours available');
        const cellContoursGeoJson = await loadCellContours();
        cellContoursObjects = getPolygonsFromMultiPolygonGeoJSON(slideMetadata, cellContoursGeoJson);
        console.log('Loaded cell contours');
      }
    }
    removeLoadingMessage();

    // Proceed with map initialization
    const cartesianProjection = new Projection({
      code: 'cartesian_projection',
      units: 'pixels',
      extent: slideMetadata.extent, // Adjust the extent as needed
      axisOrientation: 'enu',
    });

    // Tile
    const slideGrid = new TileGrid({
      extent: slideMetadata.extent,
      tileSize: [256, 256],
      minZoom: slideMetadata.minZoom,
      resolutions: slideMetadata.resolutions
    });

    // Slide
    const source = new XYZ({
      url: slideMetadata.slide_url,
      crossOrigin: 'anonymous',
      tileGrid: slideGrid,
    });

    // Define Controls
    const zoomslider = new ZoomSlider()
    const fullScreenControl = new FullScreen();
    const overviewMapControl = new OverviewMap({
      collapsed: false,
      layers: [
        new TileLayer({
          source: source,
        }),
      ],
    });
    const mousePositionControl = new MousePosition({
      coordinateFormat: function(coord) {
        const flippedY = slideMetadata.extent[3] - coord[1]; // Flip the y-coordinate
        const roundedX = Math.round(coord[0]);
        const roundedY = Math.round(flippedY);
        const mousePositionText = `[${roundedX}, ${roundedY}]`; // Format as a string and return rounded coordinates
        const mousePositionBadge = document.querySelector('.mouse-position .badge');
        if (mousePositionBadge) {
          mousePositionBadge.innerText = mousePositionText; // Update the text inside the Badge
        }
        return mousePositionText;
      },
      projection: cartesianProjection,
      undefinedHTML: '&nbsp;'
    });

    // Define map
    const map = new Map({
      target: 'map',
      layers: [
        new TileLayer({
          source: source
        }),
      ],
      projection: cartesianProjection,
      controls: defaultControls().extend([fullScreenControl, zoomslider, overviewMapControl, mousePositionControl]),

      view: new View({
        center: [slideMetadata.extent[0] + (slideMetadata.extent[2] - slideMetadata.extent[0]) / 2, slideMetadata.extent[1] + (slideMetadata.extent[3] - slideMetadata.extent[1]) / 2], // Center the map within the extent
        constrainOnlyCenter: true,
        zoom: slideMetadata.startZoom, // Starting zoom TODO:
        minZoom: slideMetadata.minZoom, // Minimum zoom level TODO:
        maxZoom: slideMetadata.maxZoom, // Maximum zoom level TODO: Check if matches to x40 or x20 in slide metadata
        extent: slideMetadata.extent
      })
    });
    fullScreenControl.setProperties({tipLabel: 'Toggle full-screen'});
    // const style = document.createElement('style');
    // document.head.appendChild(style);
    if (cellDetectionsExist || cellContoursExist) {
      // Add GUI
      const gui = new GUI();
      const gui_obj = {
        cellDetSwitch: false,
        cellContourSwitch: false
      }
      // Add cell detection layers to the map
      if (cellDetectionsExist) {
        const cellDetectionLayers = generateCellDetectionLayers(cellDetectionObjects, map);
        cellDetectionLayers.forEach(layer => map.addLayer(layer));
        gui.add(gui_obj, 'cellDetSwitch').name('Cell Detection').onChange(function(value) {
          cellDetectionLayers.forEach(layer => {
            layer.setVisible(value); // Set visibility based on the toggle switch
          });
        });
      }
      // Add cell contours layers to the map
      if (cellContoursExist) {
        const cellContoursLayers = generateCellContoursLayers(cellContoursObjects, map);
        cellContoursLayers.forEach(layer => map.addLayer(layer));
        gui.add(gui_obj, 'cellContourSwitch').name('Cell Contour').onChange(function(value) {
          cellContoursLayers.forEach(layer => {
            layer.setVisible(value); // Set visibility based on the toggle switch
          });
        });
      }
      if (cellDetectionsExist) {
        addAnnotationFolders(gui, cellDetectionObjects);
      }
      else if (cellContoursExist) {
        addAnnotationFolders(gui, cellContoursObjects);
      }
    }
  } catch (error) {
    // Handle error
    console.error('An error occurred during map initialization:', error);
    Swal.fire({
      icon: "error",
      title: "Oops...",
      text: "Something went wrong during the slide loading!",
    });
    showBackgroundMessage();
  }
}


// Main logic for the viewer

// button for upload
document.getElementById('wsi-input').addEventListener('change', function() {
  let uploadButton = document.getElementById('upload-button');
  uploadButton.disabled = this.files.length <= 0;
});

// Reset button
function fileSelected(inputElement, filenameContainerId) {
  let fileName = inputElement.files[0].name;  // Get the selected file name
  document.querySelector(`#${filenameContainerId} h6`).textContent = "Selected: " + fileName;  // Update the h6 inside the container
}
function resetPage(event) {
  event.preventDefault();

  // Clear file input values
  document.getElementById('wsi-input').value = '';
  document.getElementById('detection-input').value = '';
  document.getElementById('contour-input').value = '';

  // Reset displayed filenames
  document.querySelector('#slide-filename h6').innerHTML = "Drag and drop or select Slide";
  document.querySelector('#detection-filename h6').innerHTML = "Detection<br/>(optional)";
  document.querySelector('#contour-filename h6').innerHTML = "Contour<br/>(optional)";

  // Re-enable the upload button
  document.getElementById('upload-button').disabled = true; // Or whatever logic you have for enabling/disabling
}

// Event listener for reset button
document.addEventListener('DOMContentLoaded', () => {
  const resetButton = document.getElementById('reload-button');
  resetButton.addEventListener('click', resetPage);
});

// Event listener for file input change
document.addEventListener('DOMContentLoaded', () => {
  const resetButton = document.getElementById('reload-button');
  resetButton.addEventListener('click', resetPage);

  // Add file input change listeners
  const wsiInput = document.getElementById('wsi-input');
  const detectionInput = document.getElementById('detection-input');
  const contourInput = document.getElementById('contour-input');

  wsiInput.addEventListener('change', function() {
      fileSelected(this, 'slide-filename');
  });

  detectionInput.addEventListener('change', function() {
      fileSelected(this, 'detection-filename');
  });

  contourInput.addEventListener('change', function() {
      fileSelected(this, 'contour-filename');
  });
});


// Call the function to initialize the map
document.addEventListener('DOMContentLoaded', () => {
  let data;
  const form = document.getElementById('upload-form');
  const uploadButton = document.getElementById('upload-button');

  form.addEventListener('submit', async (event) => {
    event.preventDefault(); // Prevent form submission to allow for custom handling

    let wsiSelected = document.getElementById('wsi-input').files.length > 0;
    let detectionSelected = document.getElementById('detection-input').files.length > 0;
    let contourSelected = document.getElementById('contour-input').files.length > 0;

    // Check conditions for alert
    if (wsiSelected && (!detectionSelected && !contourSelected)) {
      const result = await Swal.fire({
        title: "No CellViT results selected",
        text: "This cannot be undone. Continue?",
        icon: "warning",
        showCancelButton: true,
        confirmButtonColor: "#3ddb62",
        cancelButtonColor: "#d33",
        confirmButtonText: "Continue",
        cancelButtonText: "Return",
      });

      if (!result.isConfirmed) {
        console.log('Form not submitted'); // User clicked cancel
        return; // Exit the function to allow further file selection
      }
    }
    console.log('Form submitted'); // User clicked OK
    const fileInput = document.getElementById('wsi-input');
    const wsiFile = fileInput.files[0];

    const detInput = document.getElementById('detection-input');
    const detFile = detInput.files[0];

    const contInput = document.getElementById('contour-input');
    const contFile = contInput.files[0];

    console.log(wsiFile);
    console.log(detFile);
    console.log(contFile);

    let upload_status = false;
    if (wsiSelected) {
      // Implement your upload logic here
      uploadButton.disabled = true; // Disable the button to prevent multiple submissions

      const formData = new FormData();
      formData.append('wsi-file', wsiFile);
      formData.append('detection-file', detFile);
      formData.append('contours-file', contFile);

      try {
        showUploadMessage();
        const response = await fetch('http://localhost:3306/upload', {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          data = await response.json();
          console.log('File uploaded successfully:', data);
          upload_status = true;
        } else {
          console.error('File upload failed:', response.statusText);
          upload_status = false;
          Swal.fire({
            icon: "error",
            title: "Oops...",
            text: "Something went wrong during the upload!",
          });
          showBackgroundMessage();
          uploadButton.disabled = false;
        }
      } catch (error) {
        console.error('There was a problem with the upload:', error);
        upload_status = false;
        Swal.fire({
          icon: "error",
          title: "Oops...",
          text: "Something went wrong during the upload!",
        });
        showBackgroundMessage();
        uploadButton.disabled = false;
      }
      console.log(data);
      if (upload_status) {
        await initializeWSIViewer(data.slide_id, data.slide_path);

        // Clear file input values
        document.getElementById('wsi-input').value = '';
        document.getElementById('detection-input').value = '';
        document.getElementById('contour-input').value = '';
        document.querySelector('#slide-filename h6').innerHTML = "Drag and drop or select slide";
        document.querySelector('#detection-filename h6').innerHTML = "Detection<br/>(optional)";
        document.querySelector('#contour-filename h6').innerHTML = "Contour<br/>(optional)";
        document.getElementById('upload-button').disabled = true; // Or whatever logic you have for enabling/disabling
      }
    }
  });

  showBackgroundMessage();
});
