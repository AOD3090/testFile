// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:srv/GetOrientationError.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_ORIENTATION_ERROR__STRUCT_HPP_
#define DSR_MSGS2__SRV__DETAIL__GET_ORIENTATION_ERROR__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__GetOrientationError_Request __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__GetOrientationError_Request __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GetOrientationError_Request_
{
  using Type = GetOrientationError_Request_<ContainerAllocator>;

  explicit GetOrientationError_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->xd.begin(), this->xd.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->xc.begin(), this->xc.end(), 0.0);
      this->axis = 0;
    }
  }

  explicit GetOrientationError_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : xd(_alloc),
    xc(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->xd.begin(), this->xd.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->xc.begin(), this->xc.end(), 0.0);
      this->axis = 0;
    }
  }

  // field types and members
  using _xd_type =
    std::array<double, 6>;
  _xd_type xd;
  using _xc_type =
    std::array<double, 6>;
  _xc_type xc;
  using _axis_type =
    int8_t;
  _axis_type axis;

  // setters for named parameter idiom
  Type & set__xd(
    const std::array<double, 6> & _arg)
  {
    this->xd = _arg;
    return *this;
  }
  Type & set__xc(
    const std::array<double, 6> & _arg)
  {
    this->xc = _arg;
    return *this;
  }
  Type & set__axis(
    const int8_t & _arg)
  {
    this->axis = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::GetOrientationError_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::GetOrientationError_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::GetOrientationError_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::GetOrientationError_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::GetOrientationError_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::GetOrientationError_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::GetOrientationError_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::GetOrientationError_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::GetOrientationError_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::GetOrientationError_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__GetOrientationError_Request
    std::shared_ptr<dsr_msgs2::srv::GetOrientationError_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__GetOrientationError_Request
    std::shared_ptr<dsr_msgs2::srv::GetOrientationError_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GetOrientationError_Request_ & other) const
  {
    if (this->xd != other.xd) {
      return false;
    }
    if (this->xc != other.xc) {
      return false;
    }
    if (this->axis != other.axis) {
      return false;
    }
    return true;
  }
  bool operator!=(const GetOrientationError_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GetOrientationError_Request_

// alias to use template instance with default allocator
using GetOrientationError_Request =
  dsr_msgs2::srv::GetOrientationError_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__GetOrientationError_Response __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__GetOrientationError_Response __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GetOrientationError_Response_
{
  using Type = GetOrientationError_Response_<ContainerAllocator>;

  explicit GetOrientationError_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->ori_error = 0.0f;
      this->success = false;
    }
  }

  explicit GetOrientationError_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->ori_error = 0.0f;
      this->success = false;
    }
  }

  // field types and members
  using _ori_error_type =
    float;
  _ori_error_type ori_error;
  using _success_type =
    bool;
  _success_type success;

  // setters for named parameter idiom
  Type & set__ori_error(
    const float & _arg)
  {
    this->ori_error = _arg;
    return *this;
  }
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::GetOrientationError_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::GetOrientationError_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::GetOrientationError_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::GetOrientationError_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::GetOrientationError_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::GetOrientationError_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::GetOrientationError_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::GetOrientationError_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::GetOrientationError_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::GetOrientationError_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__GetOrientationError_Response
    std::shared_ptr<dsr_msgs2::srv::GetOrientationError_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__GetOrientationError_Response
    std::shared_ptr<dsr_msgs2::srv::GetOrientationError_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GetOrientationError_Response_ & other) const
  {
    if (this->ori_error != other.ori_error) {
      return false;
    }
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const GetOrientationError_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GetOrientationError_Response_

// alias to use template instance with default allocator
using GetOrientationError_Response =
  dsr_msgs2::srv::GetOrientationError_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2

namespace dsr_msgs2
{

namespace srv
{

struct GetOrientationError
{
  using Request = dsr_msgs2::srv::GetOrientationError_Request;
  using Response = dsr_msgs2::srv::GetOrientationError_Response;
};

}  // namespace srv

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__GET_ORIENTATION_ERROR__STRUCT_HPP_
