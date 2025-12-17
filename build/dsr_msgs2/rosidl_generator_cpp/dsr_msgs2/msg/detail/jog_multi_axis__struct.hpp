// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:msg/JogMultiAxis.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__JOG_MULTI_AXIS__STRUCT_HPP_
#define DSR_MSGS2__MSG__DETAIL__JOG_MULTI_AXIS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__msg__JogMultiAxis __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__msg__JogMultiAxis __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct JogMultiAxis_
{
  using Type = JogMultiAxis_<ContainerAllocator>;

  explicit JogMultiAxis_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->jog_axis.begin(), this->jog_axis.end(), 0.0);
      this->move_reference = 0;
      this->speed = 0.0;
    }
  }

  explicit JogMultiAxis_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : jog_axis(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->jog_axis.begin(), this->jog_axis.end(), 0.0);
      this->move_reference = 0;
      this->speed = 0.0;
    }
  }

  // field types and members
  using _jog_axis_type =
    std::array<double, 6>;
  _jog_axis_type jog_axis;
  using _move_reference_type =
    int8_t;
  _move_reference_type move_reference;
  using _speed_type =
    double;
  _speed_type speed;

  // setters for named parameter idiom
  Type & set__jog_axis(
    const std::array<double, 6> & _arg)
  {
    this->jog_axis = _arg;
    return *this;
  }
  Type & set__move_reference(
    const int8_t & _arg)
  {
    this->move_reference = _arg;
    return *this;
  }
  Type & set__speed(
    const double & _arg)
  {
    this->speed = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::msg::JogMultiAxis_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::msg::JogMultiAxis_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::msg::JogMultiAxis_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::msg::JogMultiAxis_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::msg::JogMultiAxis_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::msg::JogMultiAxis_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::msg::JogMultiAxis_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::msg::JogMultiAxis_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::msg::JogMultiAxis_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::msg::JogMultiAxis_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__msg__JogMultiAxis
    std::shared_ptr<dsr_msgs2::msg::JogMultiAxis_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__msg__JogMultiAxis
    std::shared_ptr<dsr_msgs2::msg::JogMultiAxis_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const JogMultiAxis_ & other) const
  {
    if (this->jog_axis != other.jog_axis) {
      return false;
    }
    if (this->move_reference != other.move_reference) {
      return false;
    }
    if (this->speed != other.speed) {
      return false;
    }
    return true;
  }
  bool operator!=(const JogMultiAxis_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct JogMultiAxis_

// alias to use template instance with default allocator
using JogMultiAxis =
  dsr_msgs2::msg::JogMultiAxis_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__JOG_MULTI_AXIS__STRUCT_HPP_
