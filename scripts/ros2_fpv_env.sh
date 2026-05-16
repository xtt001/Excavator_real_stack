#!/usr/bin/env bash
# FPV / 相机启动脚本共用：主从同 ROS2 域默认值（可被环境变量覆盖）。
export EXCAVATOR_ROS2_MULTIHOST="${EXCAVATOR_ROS2_MULTIHOST:-1}"
export EXCAVATOR_ROS_DOMAIN_ID="${EXCAVATOR_ROS_DOMAIN_ID:-42}"
