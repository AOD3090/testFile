// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:msg/RobotError.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__ROBOT_ERROR__STRUCT_HPP_
#define DSR_MSGS2__MSG__DETAIL__ROBOT_ERROR__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__msg__RobotError __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__msg__RobotError __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct RobotError_
{
  using Type = RobotError_<ContainerAllocator>;

  explicit RobotError_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->level = 0l;
      this->group = 0l;
      this->code = 0l;
      this->msg1 = "";
      this->msg2 = "";
      this->msg3 = "";
    }
  }

  explicit RobotError_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : msg1(_alloc),
    msg2(_alloc),
    msg3(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->level = 0l;
      this->group = 0l;
      this->code = 0l;
      this->msg1 = "";
      this->msg2 = "";
      this->msg3 = "";
    }
  }

  // field types and members
  using _level_type =
    int32_t;
  _level_type level;
  using _group_type =
    int32_t;
  _group_type group;
  using _code_type =
    int32_t;
  _code_type code;
  using _msg1_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _msg1_type msg1;
  using _msg2_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _msg2_type msg2;
  using _msg3_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _msg3_type msg3;

  // setters for named parameter idiom
  Type & set__level(
    const int32_t & _arg)
  {
    this->level = _arg;
    return *this;
  }
  Type & set__group(
    const int32_t & _arg)
  {
    this->group = _arg;
    return *this;
  }
  Type & set__code(
    const int32_t & _arg)
  {
    this->code = _arg;
    return *this;
  }
  Type & set__msg1(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->msg1 = _arg;
    return *this;
  }
  Type & set__msg2(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->msg2 = _arg;
    return *this;
  }
  Type & set__msg3(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->msg3 = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::msg::RobotError_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::msg::RobotError_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::msg::RobotError_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::msg::RobotError_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::msg::RobotError_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::msg::RobotError_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::msg::RobotError_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::msg::RobotError_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::msg::RobotError_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::msg::RobotError_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__msg__RobotError
    std::shared_ptr<dsr_msgs2::msg::RobotError_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__msg__RobotError
    std::shared_ptr<dsr_msgs2::msg::RobotError_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const RobotError_ & other) const
  {
    if (this->level != other.level) {
      return false;
    }
    if (this->group != other.group) {
      return false;
    }
    if (this->code != other.code) {
      return false;
    }
    if (this->msg1 != other.msg1) {
      return false;
    }
    if (this->msg2 != other.msg2) {
      return false;
    }
    if (this->msg3 != other.msg3) {
      return false;
    }
    return true;
  }
  bool operator!=(const RobotError_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct RobotError_

// alias to use template instance with default allocator
using RobotError =
  dsr_msgs2::msg::RobotError_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__ROBOT_ERROR__STRUCT_HPP_
