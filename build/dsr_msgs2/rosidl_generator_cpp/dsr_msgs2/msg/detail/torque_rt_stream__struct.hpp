// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:msg/TorqueRtStream.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__TORQUE_RT_STREAM__STRUCT_HPP_
#define DSR_MSGS2__MSG__DETAIL__TORQUE_RT_STREAM__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__msg__TorqueRtStream __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__msg__TorqueRtStream __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct TorqueRtStream_
{
  using Type = TorqueRtStream_<ContainerAllocator>;

  explicit TorqueRtStream_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->tor.begin(), this->tor.end(), 0.0);
      this->time = 0.0;
    }
  }

  explicit TorqueRtStream_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : tor(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->tor.begin(), this->tor.end(), 0.0);
      this->time = 0.0;
    }
  }

  // field types and members
  using _tor_type =
    std::array<double, 6>;
  _tor_type tor;
  using _time_type =
    double;
  _time_type time;

  // setters for named parameter idiom
  Type & set__tor(
    const std::array<double, 6> & _arg)
  {
    this->tor = _arg;
    return *this;
  }
  Type & set__time(
    const double & _arg)
  {
    this->time = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::msg::TorqueRtStream_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::msg::TorqueRtStream_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::msg::TorqueRtStream_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::msg::TorqueRtStream_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::msg::TorqueRtStream_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::msg::TorqueRtStream_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::msg::TorqueRtStream_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::msg::TorqueRtStream_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::msg::TorqueRtStream_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::msg::TorqueRtStream_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__msg__TorqueRtStream
    std::shared_ptr<dsr_msgs2::msg::TorqueRtStream_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__msg__TorqueRtStream
    std::shared_ptr<dsr_msgs2::msg::TorqueRtStream_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const TorqueRtStream_ & other) const
  {
    if (this->tor != other.tor) {
      return false;
    }
    if (this->time != other.time) {
      return false;
    }
    return true;
  }
  bool operator!=(const TorqueRtStream_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct TorqueRtStream_

// alias to use template instance with default allocator
using TorqueRtStream =
  dsr_msgs2::msg::TorqueRtStream_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__TORQUE_RT_STREAM__STRUCT_HPP_
