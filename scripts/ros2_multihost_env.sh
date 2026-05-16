#!/usr/bin/env bash
# 主从同 ROS2 域（DDS 跨机）。须在 source /opt/ros/... 之前执行。
# 用法: source scripts/ros2_multihost_env.sh

_multihost_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_dds_dir="${_multihost_root}/configs/ros2_dds"
_domain_id="${EXCAVATOR_ROS_DOMAIN_ID:-42}"
_rmw="${EXCAVATOR_RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

export ROS_DOMAIN_ID="${_domain_id}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION="${_rmw}"

_peer_ip="${EXCAVATOR_ROS_PEER_IP:-}"
if [[ -n "${_peer_ip}" ]]; then
  _gen="${_dds_dir}/.cyclonedds_peers_${_peer_ip//./_}.xml"
  sed "s/@EXCAVATOR_ROS_PEER_IP@/${_peer_ip}/g" \
    "${_dds_dir}/cyclonedds_multihost_peers.xml.in" > "${_gen}"
  export CYCLONEDDS_URI="file://${_gen}"
else
  export CYCLONEDDS_URI="file://${_dds_dir}/cyclonedds_multihost.xml"
fi

echo "ros2_multihost: DOMAIN_ID=${ROS_DOMAIN_ID} RMW=${RMW_IMPLEMENTATION}" >&2
echo "ros2_multihost: CYCLONEDDS_URI=${CYCLONEDDS_URI}" >&2
if [[ -n "${_peer_ip}" ]]; then
  echo "ros2_multihost: peer=${_peer_ip}" >&2
else
  echo "ros2_multihost: multicast discovery (set EXCAVATOR_ROS_PEER_IP if topics missing)" >&2
fi
