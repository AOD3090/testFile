// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:msg/LogAlarm.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__LOG_ALARM__STRUCT_HPP_
#define DSR_MSGS2__MSG__DETAIL__LOG_ALARM__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__msg__LogAlarm __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__msg__LogAlarm __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct LogAlarm_
{
  using Type = LogAlarm_<ContainerAllocator>;

  explicit LogAlarm_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->level = 0l;
      this->group = 0l;
      this->index = 0l;
      std::fill<typename std::array<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>, 3>::iterator, std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>>(this->param.begin(), this->param.end(), "");
    }
  }

  explicit LogAlarm_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : param(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->level = 0l;
      this->group = 0l;
      this->index = 0l;
      std::fill<typename std::array<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>, 3>::iterator, std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>>(this->param.begin(), this->param.end(), "");
    }
  }

  // field types and members
  using _level_type =
    int32_t;
  _level_type level;
  using _group_type =
    int32_t;
  _group_type group;
  using _index_type =
    int32_t;
  _index_type index;
  using _param_type =
    std::array<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>, 3>;
  _param_type param;

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
  Type & set__index(
    const int32_t & _arg)
  {
    this->index = _arg;
    return *this;
  }
  Type & set__param(
    const std::array<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>, 3> & _arg)
  {
    this->param = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::msg::LogAlarm_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::msg::LogAlarm_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::msg::LogAlarm_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::msg::LogAlarm_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::msg::LogAlarm_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::msg::LogAlarm_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::msg::LogAlarm_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::msg::LogAlarm_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::msg::LogAlarm_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::msg::LogAlarm_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__msg__LogAlarm
    std::shared_ptr<dsr_msgs2::msg::LogAlarm_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__msg__LogAlarm
    std::shared_ptr<dsr_msgs2::msg::LogAlarm_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const LogAlarm_ & other) const
  {
    if (this->level != other.level) {
      return false;
    }
    if (this->group != other.group) {
      return false;
    }
    if (this->index != other.index) {
      return false;
    }
    if (this->param != other.param) {
      return false;
    }
    return true;
  }
  bool operator!=(const LogAlarm_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct LogAlarm_

// alias to use template instance with default allocator
using LogAlarm =
  dsr_msgs2::msg::LogAlarm_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__LOG_ALARM__STRUCT_HPP_
