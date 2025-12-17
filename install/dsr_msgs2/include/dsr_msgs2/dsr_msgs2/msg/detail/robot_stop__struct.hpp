// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:msg/RobotStop.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__ROBOT_STOP__STRUCT_HPP_
#define DSR_MSGS2__MSG__DETAIL__ROBOT_STOP__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__msg__RobotStop __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__msg__RobotStop __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct RobotStop_
{
  using Type = RobotStop_<ContainerAllocator>;

  explicit RobotStop_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->stop_mode = 0l;
    }
  }

  explicit RobotStop_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->stop_mode = 0l;
    }
  }

  // field types and members
  using _stop_mode_type =
    int32_t;
  _stop_mode_type stop_mode;

  // setters for named parameter idiom
  Type & set__stop_mode(
    const int32_t & _arg)
  {
    this->stop_mode = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::msg::RobotStop_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::msg::RobotStop_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::msg::RobotStop_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::msg::RobotStop_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::msg::RobotStop_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::msg::RobotStop_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::msg::RobotStop_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::msg::RobotStop_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::msg::RobotStop_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::msg::RobotStop_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__msg__RobotStop
    std::shared_ptr<dsr_msgs2::msg::RobotStop_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__msg__RobotStop
    std::shared_ptr<dsr_msgs2::msg::RobotStop_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const RobotStop_ & other) const
  {
    if (this->stop_mode != other.stop_mode) {
      return false;
    }
    return true;
  }
  bool operator!=(const RobotStop_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct RobotStop_

// alias to use template instance with default allocator
using RobotStop =
  dsr_msgs2::msg::RobotStop_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__ROBOT_STOP__STRUCT_HPP_
