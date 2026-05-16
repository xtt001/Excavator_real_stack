#!/usr/bin/env bash
# 主从同 ROS2 域。须在 source /opt/ros/... 之前执行。
# 用法: source scripts/ros2_multihost_env.sh

_multihost_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_dds_dir="${_multihost_root}/configs/ros2_dds"
_domain_id="${EXCAVATOR_ROS_DOMAIN_ID:-42}"

export ROS_DOMAIN_ID="${_domain_id}"
export ROS_LOCALHOST_ONLY=0

_pick_rmw() {
  if [[ -n "${EXCAVATOR_RMW_IMPLEMENTATION:-}" ]]; then
    echo "${EXCAVATOR_RMW_IMPLEMENTATION}"
    return
  fi
  local ros_lib="/opt/ros/${ROS_DISTRO:-humble}/lib"
  if [[ -f "${ros_lib}/librmw_cyclonedds_cpp.so" ]]; then
    echo "rmw_cyclonedds_cpp"
  else
    echo "rmw_fastrtps_cpp"
  fi
}

_rmw="$(_pick_rmw)"
export RMW_IMPLEMENTATION="${_rmw}"

if [[ "${_rmw}" == "rmw_cyclonedds_cpp" ]]; then
  _peer_ip="${EXCAVATOR_ROS_PEER_IP:-}"
  if [[ -n "${_peer_ip}" ]]; then
    _gen="${_dds_dir}/.cyclonedds_peers_${_peer_ip//./_}.xml"
    sed "s/@EXCAVATOR_ROS_PEER_IP@/${_peer_ip}/g" \
      "${_dds_dir}/cyclonedds_multihost_peers.xml.in" > "${_gen}"
    export CYCLONEDDS_URI="file://${_gen}"
  else
    export CYCLONEDDS_URI="file://${_dds_dir}/cyclonedds_multihost.xml"
  fi
else
  unset CYCLONEDDS_URI
fi

echo "ros2_multihost: DOMAIN_ID=${ROS_DOMAIN_ID} RMW=${RMW_IMPLEMENTATION}" >&2
if [[ "${_rmw}" == "rmw_cyclonedds_cpp" ]]; then
  echo "ros2_multihost: CYCLONEDDS_URI=${CYCLONEDDS_URI}" >&2
else
  echo "ros2_multihost: Fast DDS (install ros-humble-rmw-cyclonedds-cpp for Cyclone)" >&2
fi
if [[ -n "${EXCAVATOR_ROS_PEER_IP:-}" ]]; then
  echo "ros2_multihost: peer=${EXCAVATOR_ROS_PEER_IP}" >&2
fi
