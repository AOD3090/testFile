// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:srv/SetCtrlBoxAnalogInputType.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_CTRL_BOX_ANALOG_INPUT_TYPE__STRUCT_HPP_
#define DSR_MSGS2__SRV__DETAIL__SET_CTRL_BOX_ANALOG_INPUT_TYPE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__SetCtrlBoxAnalogInputType_Request __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__SetCtrlBoxAnalogInputType_Request __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetCtrlBoxAnalogInputType_Request_
{
  using Type = SetCtrlBoxAnalogInputType_Request_<ContainerAllocator>;

  explicit SetCtrlBoxAnalogInputType_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->channel = 0;
      this->mode = 0;
    }
  }

  explicit SetCtrlBoxAnalogInputType_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->channel = 0;
      this->mode = 0;
    }
  }

  // field types and members
  using _channel_type =
    int8_t;
  _channel_type channel;
  using _mode_type =
    int8_t;
  _mode_type mode;

  // setters for named parameter idiom
  Type & set__channel(
    const int8_t & _arg)
  {
    this->channel = _arg;
    return *this;
  }
  Type & set__mode(
    const int8_t & _arg)
  {
    this->mode = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__SetCtrlBoxAnalogInputType_Request
    std::shared_ptr<dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__SetCtrlBoxAnalogInputType_Request
    std::shared_ptr<dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetCtrlBoxAnalogInputType_Request_ & other) const
  {
    if (this->channel != other.channel) {
      return false;
    }
    if (this->mode != other.mode) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetCtrlBoxAnalogInputType_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetCtrlBoxAnalogInputType_Request_

// alias to use template instance with default allocator
using SetCtrlBoxAnalogInputType_Request =
  dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__SetCtrlBoxAnalogInputType_Response __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__SetCtrlBoxAnalogInputType_Response __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetCtrlBoxAnalogInputType_Response_
{
  using Type = SetCtrlBoxAnalogInputType_Response_<ContainerAllocator>;

  explicit SetCtrlBoxAnalogInputType_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit SetCtrlBoxAnalogInputType_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__SetCtrlBoxAnalogInputType_Response
    std::shared_ptr<dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__SetCtrlBoxAnalogInputType_Response
    std::shared_ptr<dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetCtrlBoxAnalogInputType_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetCtrlBoxAnalogInputType_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetCtrlBoxAnalogInputType_Response_

// alias to use template instance with default allocator
using SetCtrlBoxAnalogInputType_Response =
  dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2

namespace dsr_msgs2
{

namespace srv
{

struct SetCtrlBoxAnalogInputType
{
  using Request = dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Request;
  using Response = dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Response;
};

}  // namespace srv

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__SET_CTRL_BOX_ANALOG_INPUT_TYPE__STRUCT_HPP_
