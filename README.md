# Robotic Arm Gas Source Localization

A bio-inspired gas source localization framework for a 7-DoF robotic arm using ROS2 simulation and a single-point gas sensor mounted on the end effector.

## Overview

This repository contains the core code for a robotic arm gas source localization project.  
The system combines gas dispersion simulation, gas sensor simulation, and a bio-inspired search controller to locate a gas source in a simulated environment.

## Main Components

- `gas_seek_bio.py`  
  Main search algorithm for gas source localization.

- `gas_dispersion_simulator.py`  
  Simulation of gas dispersion in the environment.

- `gas_sensor_simulator.py`  
  Simulation of gas sensor response.

- `nodegas.py`  
  ROS2-related gas simulation node.

## Project Goal

The goal of this project is to investigate how a 7-DoF robotic arm equipped with a single-point gas sensor mounted on its end effector can perform gas source localization using bio-inspired search strategies.

## Notes

This repository is organized as part of a bachelor thesis project on robotic arm gas source localization.
