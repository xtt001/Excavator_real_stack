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
  _bind_ip="${EXCAVATOR_ROS_BIND_IP:-}"
  if [[ -z "${_bind_ip}" && -n "${_peer_ip}" ]] && command -v ip >/dev/null 2>&1; then
    _bind_ip="$(
      ip -4 route get "${_peer_ip}" 2>/dev/null \
        | awk '{for (i = 1; i <= NF; ++i) if ($i == "src") {print $(i + 1); exit}}'
    )"
  fi
  if [[ -n "${_peer_ip}" ]]; then
    _bind_tag="${_bind_ip:-auto}"
    _gen="${_dds_dir}/.cyclonedds_bind_${_bind_tag//./_}_peer_${_peer_ip//./_}.xml"
    {
      cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!-- 组播不可靠时用：两端各设 EXCAVATOR_ROS_PEER_IP=对方 IP。 -->
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain>
    <General>
      <NetworkInterfaceAddress>${_bind_tag}</NetworkInterfaceAddress>
      <AllowMulticast>false</AllowMulticast>
    </General>
    <Discovery>
      <Peers>
        <Peer address="${_peer_ip}"/>
EOF
      if [[ -n "${_bind_ip}" && "${_bind_ip}" != "${_peer_ip}" ]]; then
        echo "        <Peer address=\"${_bind_ip}\"/>"
      fi
      cat <<EOF
      </Peers>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>120</MaxAutoParticipantIndex>
    </Discovery>
  </Domain>
</CycloneDDS>
EOF
    } > "${_gen}"
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
  echo "ros2_multihost: Fast DDS (install ros-${ROS_DISTRO:-humble}-rmw-cyclonedds-cpp for Cyclone)" >&2
fi
if [[ -n "${EXCAVATOR_ROS_PEER_IP:-}" ]]; then
  echo "ros2_multihost: peer=${EXCAVATOR_ROS_PEER_IP}" >&2
fi
if [[ -n "${EXCAVATOR_ROS_BIND_IP:-}" ]]; then
  echo "ros2_multihost: bind=${EXCAVATOR_ROS_BIND_IP}" >&2
fi
