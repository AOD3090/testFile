// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:msg/AlterMotionStream.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__ALTER_MOTION_STREAM__STRUCT_HPP_
#define DSR_MSGS2__MSG__DETAIL__ALTER_MOTION_STREAM__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__msg__AlterMotionStream __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__msg__AlterMotionStream __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct AlterMotionStream_
{
  using Type = AlterMotionStream_<ContainerAllocator>;

  explicit AlterMotionStream_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->pos.begin(), this->pos.end(), 0.0);
    }
  }

  explicit AlterMotionStream_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : pos(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->pos.begin(), this->pos.end(), 0.0);
    }
  }

  // field types and members
  using _pos_type =
    std::array<double, 6>;
  _pos_type pos;

  // setters for named parameter idiom
  Type & set__pos(
    const std::array<double, 6> & _arg)
  {
    this->pos = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::msg::AlterMotionStream_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::msg::AlterMotionStream_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::msg::AlterMotionStream_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::msg::AlterMotionStream_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::msg::AlterMotionStream_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::msg::AlterMotionStream_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::msg::AlterMotionStream_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::msg::AlterMotionStream_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::msg::AlterMotionStream_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::msg::AlterMotionStream_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__msg__AlterMotionStream
    std::shared_ptr<dsr_msgs2::msg::AlterMotionStream_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__msg__AlterMotionStream
    std::shared_ptr<dsr_msgs2::msg::AlterMotionStream_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const AlterMotionStream_ & other) const
  {
    if (this->pos != other.pos) {
      return false;
    }
    return true;
  }
  bool operator!=(const AlterMotionStream_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct AlterMotionStream_

// alias to use template instance with default allocator
using AlterMotionStream =
  dsr_msgs2::msg::AlterMotionStream_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__ALTER_MOTION_STREAM__STRUCT_HPP_
