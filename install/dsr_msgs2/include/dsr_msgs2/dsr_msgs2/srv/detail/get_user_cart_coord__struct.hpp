// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:srv/GetUserCartCoord.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_USER_CART_COORD__STRUCT_HPP_
#define DSR_MSGS2__SRV__DETAIL__GET_USER_CART_COORD__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__GetUserCartCoord_Request __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__GetUserCartCoord_Request __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GetUserCartCoord_Request_
{
  using Type = GetUserCartCoord_Request_<ContainerAllocator>;

  explicit GetUserCartCoord_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id = 0;
    }
  }

  explicit GetUserCartCoord_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id = 0;
    }
  }

  // field types and members
  using _id_type =
    int8_t;
  _id_type id;

  // setters for named parameter idiom
  Type & set__id(
    const int8_t & _arg)
  {
    this->id = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::GetUserCartCoord_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::GetUserCartCoord_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::GetUserCartCoord_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::GetUserCartCoord_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::GetUserCartCoord_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::GetUserCartCoord_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::GetUserCartCoord_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::GetUserCartCoord_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::GetUserCartCoord_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::GetUserCartCoord_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__GetUserCartCoord_Request
    std::shared_ptr<dsr_msgs2::srv::GetUserCartCoord_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__GetUserCartCoord_Request
    std::shared_ptr<dsr_msgs2::srv::GetUserCartCoord_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GetUserCartCoord_Request_ & other) const
  {
    if (this->id != other.id) {
      return false;
    }
    return true;
  }
  bool operator!=(const GetUserCartCoord_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GetUserCartCoord_Request_

// alias to use template instance with default allocator
using GetUserCartCoord_Request =
  dsr_msgs2::srv::GetUserCartCoord_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__GetUserCartCoord_Response __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__GetUserCartCoord_Response __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GetUserCartCoord_Response_
{
  using Type = GetUserCartCoord_Response_<ContainerAllocator>;

  explicit GetUserCartCoord_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->conv_posx.begin(), this->conv_posx.end(), 0.0);
      this->ref = 0;
      this->success = false;
    }
  }

  explicit GetUserCartCoord_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : conv_posx(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->conv_posx.begin(), this->conv_posx.end(), 0.0);
      this->ref = 0;
      this->success = false;
    }
  }

  // field types and members
  using _conv_posx_type =
    std::array<double, 6>;
  _conv_posx_type conv_posx;
  using _ref_type =
    int8_t;
  _ref_type ref;
  using _success_type =
    bool;
  _success_type success;

  // setters for named parameter idiom
  Type & set__conv_posx(
    const std::array<double, 6> & _arg)
  {
    this->conv_posx = _arg;
    return *this;
  }
  Type & set__ref(
    const int8_t & _arg)
  {
    this->ref = _arg;
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
    dsr_msgs2::srv::GetUserCartCoord_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::GetUserCartCoord_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::GetUserCartCoord_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::GetUserCartCoord_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::GetUserCartCoord_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::GetUserCartCoord_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::GetUserCartCoord_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::GetUserCartCoord_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::GetUserCartCoord_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::GetUserCartCoord_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__GetUserCartCoord_Response
    std::shared_ptr<dsr_msgs2::srv::GetUserCartCoord_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__GetUserCartCoord_Response
    std::shared_ptr<dsr_msgs2::srv::GetUserCartCoord_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GetUserCartCoord_Response_ & other) const
  {
    if (this->conv_posx != other.conv_posx) {
      return false;
    }
    if (this->ref != other.ref) {
      return false;
    }
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const GetUserCartCoord_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GetUserCartCoord_Response_

// alias to use template instance with default allocator
using GetUserCartCoord_Response =
  dsr_msgs2::srv::GetUserCartCoord_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2

namespace dsr_msgs2
{

namespace srv
{

struct GetUserCartCoord
{
  using Request = dsr_msgs2::srv::GetUserCartCoord_Request;
  using Response = dsr_msgs2::srv::GetUserCartCoord_Response;
};

}  // namespace srv

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__GET_USER_CART_COORD__STRUCT_HPP_
