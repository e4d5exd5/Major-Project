metadata = {
  "IP": {
    "name": "indian_pines",
    "data": {
      "suffix": "_corrected",
      "ext": ".mat",
      "key": "indian_pines_corrected"
    },
    "label": {
      "suffix": "_gt",
      "ext": ".mat",
      "key": "indian_pines_gt"
    },
    "target_names": [
      "Unclassified",
      "Alfalfa",
      "Corn-notill",
      "Corn-mintill",
      "Corn",
      "Grass-pasture",
      "Grass-trees",
      "Grass-pasture-mowed",
      "Hay-windrowed",
      "Oats",
      "Soybean-notill",
      "Soybean-mintill",
      "Soybean-clean",
      "Wheat",
      "Woods",
      "Buildings-Grass-Trees-Drives",
      "Stone-Steel-Towers"
    ],
    "num_classes": 16,
    # "training_classes": [0, 2, 3, 4, 7, 10, 11, 12, 14, 15],
    # "testing_classes": [1, 5, 6, 8, 9, 13],
    # "training_classes": [1, 2, 4, 5, 7, 9, 10, 11, 13, 14],
    # "testing_classes": [0, 3, 6, 8, 12, 15],
    "training_classes": [0, 2, 3, 4, 5, 7, 10, 11, 12, 15],
    "testing_classes": [1, 6, 8, 9, 13, 14],
    # (0, 2, 3, 4, 5, 7, 10, 11, 12, 15)","(1, 6, 8, 9, 13, 14)
    "label_offset": 1
  },
  


  "SA": {
    "name": "salinas",
    "data": {
      "suffix": "_corrected",
      "ext": ".mat",
      "key": "salinas_corrected"
    },
    "label": {
      "suffix": "_gt",
      "ext": ".mat",
      "key": "salinas_gt"
    },
    "target_names": [
      "Unclassified",
      "Brocoli_green_weeds_1",
      "Brocoli_green_weeds_2",
      "Fallow",
      "Fallow_rough_plow",
      "Fallow_smooth",
      "Stubble",
      "Celery",
      "Grapes_untrained",
      "Soil_vinyard_develop",
      "Corn_senesced_green_weeds",
      "Lettuce_romaine_4wk",
      "Lettuce_romaine_5wk",
      "Lettuce_romaine_6wk",
      "Lettuce_romaine_7wk",
      "Vinyard_untrained",
      "Vinyard_vertical_trellis"
    ],
    "num_classes": 16,
    "training_classes": [1, 2, 4, 5, 7, 9, 10, 11, 13, 14],
    "testing_classes": [0, 3, 6, 8, 12, 15],
    "label_offset": 1
  },
  
  "PU": {
    "name": "PaviaU",
    "data": {
      "suffix": "",
      "ext": ".mat",
      "key": "paviaU"
    },
    "label": {
      "suffix": "_gt",
      "ext": ".mat",
      "key": "paviaU_gt"
    },
    "target_names": [
      "Unclassified",
      "Asphalt",
      "Meadows",
      "Gravel",
      "Trees",
      "Painted metal sheets",
      "Bare Soil",
      "Bitumen",
      "Self-Blocking Bricks",
      "Shadows"
    ],
    "num_classes": 9,
    "training_classes": [1, 2, 4, 5, 7, 8],
    "testing_classes": [0, 3, 6],
    "label_offset": 1
  },
  
  "HU": {
    "name": "Houston",
    "data": {
      "suffix": "",
      "ext": ".mat",
      "key": "houston"
    },
    "label": {
      "suffix": "_gt",
      "ext": ".mat",
      "key": "houston_gt_tr"
    },
    "target_names": [
      "Unclassified",
      "Healthy grass",
      "Stressed grass",
      "Synthetic grass",
      "Trees",
      "Soil",
      "Water",
      "Residential",
      "Commercial",
      "Road",
      "Highway",
      "Railway",
      "Parking Lot 1",
      "Parking Lot 2",
      "Tennis Court",
      "Running Track"
    ],
    "num_classes": 15,
    "training_classes": [0, 2, 4, 6, 8, 10, 12, 13, 14],
    "testing_classes": [1, 3, 5, 7, 9, 11],
    "label_offset": 0
  },
  "BW": {
      "name": "Botswana",
      "data": {
          "suffix": "",
          "ext": ".mat",
          "key": "Botswana"
      },
      "label": {
          "suffix": "_gt",
          "ext": ".mat",
          "key": "Botswana_gt"
      },
      "target_names": [
          "Unclassified",
          "Water",
          "Hippo grass",
          "floodplain grasses 1",
          "floodplain grasses 2",
          "reedsl",
          "riparian",
          "firescar2",
          "island interior",
          "acacia woodlands",
          "acacia shrublands",
          "acacia grasslands",
          "short mopane",
          "mixed mopane",
          "exposed soils"
      ],
      "num_classes": 14,
      "training_classes": [1, 2, 4, 5, 7, 9, 10, 11, 13],
      "testing_classes": [0, 3, 6, 8, 12],
      "label_offset": 1
  },
  "KSC": {
      "name": "KSC",
      "data": {
          "suffix": "_corrected",
          "ext": ".mat",
          "key": "KSC"
      },
      "label": {
          "suffix": "_gt",
          "ext": ".mat",
          "key": "KSC_gt"
      },
      "target_names": [
          "Unclassified",
          "Scrub",
          "Willow Swamp",
          "Cabbage Palm Hammock",
          "Cabbage Palm/Oak Hammock",
          "Slash Pine",
          "Oak/Broadleaf Hammock",
          "Hardwood Swamp",
          "Graminoid Marsh",
          "Spartina Marsh",
          "Cattail Marsh",
          "Salt Marsh",
          "Mud Flats",
          "Water"

      ],
      "num_classes": 13,
      "training_classes": [1, 2, 4, 5, 7, 9, 10, 11],
      "testing_classes": [0, 3, 6, 8, 12],
      "label_offset": 1
  },
  
}
