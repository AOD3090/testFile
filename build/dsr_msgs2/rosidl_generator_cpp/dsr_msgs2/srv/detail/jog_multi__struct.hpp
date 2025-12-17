// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:srv/JogMulti.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__JOG_MULTI__STRUCT_HPP_
#define DSR_MSGS2__SRV__DETAIL__JOG_MULTI__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__JogMulti_Request __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__JogMulti_Request __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct JogMulti_Request_
{
  using Type = JogMulti_Request_<ContainerAllocator>;

  explicit JogMulti_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->jog_axis.begin(), this->jog_axis.end(), 0.0);
      this->move_reference = 0;
      this->speed = 0.0;
    }
  }

  explicit JogMulti_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    dsr_msgs2::srv::JogMulti_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::JogMulti_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::JogMulti_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::JogMulti_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::JogMulti_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::JogMulti_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::JogMulti_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::JogMulti_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::JogMulti_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::JogMulti_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__JogMulti_Request
    std::shared_ptr<dsr_msgs2::srv::JogMulti_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__JogMulti_Request
    std::shared_ptr<dsr_msgs2::srv::JogMulti_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const JogMulti_Request_ & other) const
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
  bool operator!=(const JogMulti_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct JogMulti_Request_

// alias to use template instance with default allocator
using JogMulti_Request =
  dsr_msgs2::srv::JogMulti_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__JogMulti_Response __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__JogMulti_Response __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct JogMulti_Response_
{
  using Type = JogMulti_Response_<ContainerAllocator>;

  explicit JogMulti_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit JogMulti_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  // field types and members
  using _success_type =
    bool;
  _success_type success;

  // setters for named parameter idiom
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::JogMulti_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::JogMulti_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::JogMulti_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::JogMulti_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::JogMulti_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::JogMulti_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::JogMulti_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::JogMulti_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::JogMulti_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::JogMulti_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__JogMulti_Response
    std::shared_ptr<dsr_msgs2::srv::JogMulti_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__JogMulti_Response
    std::shared_ptr<dsr_msgs2::srv::JogMulti_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const JogMulti_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const JogMulti_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct JogMulti_Response_

// alias to use template instance with default allocator
using JogMulti_Response =
  dsr_msgs2::srv::JogMulti_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2

namespace dsr_msgs2
{

namespace srv
{

struct JogMulti
{
  using Request = dsr_msgs2::srv::JogMulti_Request;
  using Response = dsr_msgs2::srv::JogMulti_Response;
};

}  // namespace srv

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__JOG_MULTI__STRUCT_HPP_
