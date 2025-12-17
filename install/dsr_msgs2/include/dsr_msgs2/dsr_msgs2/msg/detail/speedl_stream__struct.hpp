// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:msg/SpeedlStream.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__SPEEDL_STREAM__STRUCT_HPP_
#define DSR_MSGS2__MSG__DETAIL__SPEEDL_STREAM__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__msg__SpeedlStream __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__msg__SpeedlStream __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct SpeedlStream_
{
  using Type = SpeedlStream_<ContainerAllocator>;

  explicit SpeedlStream_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->vel.begin(), this->vel.end(), 0.0);
      std::fill<typename std::array<double, 2>::iterator, double>(this->acc.begin(), this->acc.end(), 0.0);
      this->time = 0.0;
    }
  }

  explicit SpeedlStream_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : vel(_alloc),
    acc(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->vel.begin(), this->vel.end(), 0.0);
      std::fill<typename std::array<double, 2>::iterator, double>(this->acc.begin(), this->acc.end(), 0.0);
      this->time = 0.0;
    }
  }

  // field types and members
  using _vel_type =
    std::array<double, 6>;
  _vel_type vel;
  using _acc_type =
    std::array<double, 2>;
  _acc_type acc;
  using _time_type =
    double;
  _time_type time;

  // setters for named parameter idiom
  Type & set__vel(
    const std::array<double, 6> & _arg)
  {
    this->vel = _arg;
    return *this;
  }
  Type & set__acc(
    const std::array<double, 2> & _arg)
  {
    this->acc = _arg;
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
    dsr_msgs2::msg::SpeedlStream_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::msg::SpeedlStream_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::msg::SpeedlStream_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::msg::SpeedlStream_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::msg::SpeedlStream_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::msg::SpeedlStream_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::msg::SpeedlStream_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::msg::SpeedlStream_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::msg::SpeedlStream_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::msg::SpeedlStream_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__msg__SpeedlStream
    std::shared_ptr<dsr_msgs2::msg::SpeedlStream_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__msg__SpeedlStream
    std::shared_ptr<dsr_msgs2::msg::SpeedlStream_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SpeedlStream_ & other) const
  {
    if (this->vel != other.vel) {
      return false;
    }
    if (this->acc != other.acc) {
      return false;
    }
    if (this->time != other.time) {
      return false;
    }
    return true;
  }
  bool operator!=(const SpeedlStream_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SpeedlStream_

// alias to use template instance with default allocator
using SpeedlStream =
  dsr_msgs2::msg::SpeedlStream_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__SPEEDL_STREAM__STRUCT_HPP_
