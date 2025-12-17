// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:srv/SetUserCartCoord3.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_USER_CART_COORD3__STRUCT_HPP_
#define DSR_MSGS2__SRV__DETAIL__SET_USER_CART_COORD3__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__SetUserCartCoord3_Request __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__SetUserCartCoord3_Request __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetUserCartCoord3_Request_
{
  using Type = SetUserCartCoord3_Request_<ContainerAllocator>;

  explicit SetUserCartCoord3_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 3>::iterator, double>(this->u1.begin(), this->u1.end(), 0.0);
      std::fill<typename std::array<double, 3>::iterator, double>(this->v1.begin(), this->v1.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->pos.begin(), this->pos.end(), 0.0);
      this->ref = 0;
    }
  }

  explicit SetUserCartCoord3_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : u1(_alloc),
    v1(_alloc),
    pos(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 3>::iterator, double>(this->u1.begin(), this->u1.end(), 0.0);
      std::fill<typename std::array<double, 3>::iterator, double>(this->v1.begin(), this->v1.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->pos.begin(), this->pos.end(), 0.0);
      this->ref = 0;
    }
  }

  // field types and members
  using _u1_type =
    std::array<double, 3>;
  _u1_type u1;
  using _v1_type =
    std::array<double, 3>;
  _v1_type v1;
  using _pos_type =
    std::array<double, 6>;
  _pos_type pos;
  using _ref_type =
    int8_t;
  _ref_type ref;

  // setters for named parameter idiom
  Type & set__u1(
    const std::array<double, 3> & _arg)
  {
    this->u1 = _arg;
    return *this;
  }
  Type & set__v1(
    const std::array<double, 3> & _arg)
  {
    this->v1 = _arg;
    return *this;
  }
  Type & set__pos(
    const std::array<double, 6> & _arg)
  {
    this->pos = _arg;
    return *this;
  }
  Type & set__ref(
    const int8_t & _arg)
  {
    this->ref = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::SetUserCartCoord3_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::SetUserCartCoord3_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::SetUserCartCoord3_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::SetUserCartCoord3_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::SetUserCartCoord3_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::SetUserCartCoord3_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::SetUserCartCoord3_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::SetUserCartCoord3_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::SetUserCartCoord3_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::SetUserCartCoord3_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__SetUserCartCoord3_Request
    std::shared_ptr<dsr_msgs2::srv::SetUserCartCoord3_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__SetUserCartCoord3_Request
    std::shared_ptr<dsr_msgs2::srv::SetUserCartCoord3_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetUserCartCoord3_Request_ & other) const
  {
    if (this->u1 != other.u1) {
      return false;
    }
    if (this->v1 != other.v1) {
      return false;
    }
    if (this->pos != other.pos) {
      return false;
    }
    if (this->ref != other.ref) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetUserCartCoord3_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetUserCartCoord3_Request_

// alias to use template instance with default allocator
using SetUserCartCoord3_Request =
  dsr_msgs2::srv::SetUserCartCoord3_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__SetUserCartCoord3_Response __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__SetUserCartCoord3_Response __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetUserCartCoord3_Response_
{
  using Type = SetUserCartCoord3_Response_<ContainerAllocator>;

  explicit SetUserCartCoord3_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id = 0;
      this->success = false;
    }
  }

  explicit SetUserCartCoord3_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id = 0;
      this->success = false;
    }
  }

  // field types and members
  using _id_type =
    int8_t;
  _id_type id;
  using _success_type =
    bool;
  _success_type success;

  // setters for named parameter idiom
  Type & set__id(
    const int8_t & _arg)
  {
    this->id = _arg;
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
    dsr_msgs2::srv::SetUserCartCoord3_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::SetUserCartCoord3_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::SetUserCartCoord3_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::SetUserCartCoord3_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::SetUserCartCoord3_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::SetUserCartCoord3_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::SetUserCartCoord3_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::SetUserCartCoord3_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::SetUserCartCoord3_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::SetUserCartCoord3_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__SetUserCartCoord3_Response
    std::shared_ptr<dsr_msgs2::srv::SetUserCartCoord3_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__SetUserCartCoord3_Response
    std::shared_ptr<dsr_msgs2::srv::SetUserCartCoord3_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetUserCartCoord3_Response_ & other) const
  {
    if (this->id != other.id) {
      return false;
    }
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetUserCartCoord3_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetUserCartCoord3_Response_

// alias to use template instance with default allocator
using SetUserCartCoord3_Response =
  dsr_msgs2::srv::SetUserCartCoord3_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2

namespace dsr_msgs2
{

namespace srv
{

struct SetUserCartCoord3
{
  using Request = dsr_msgs2::srv::SetUserCartCoord3_Request;
  using Response = dsr_msgs2::srv::SetUserCartCoord3_Response;
};

}  // namespace srv

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__SET_USER_CART_COORD3__STRUCT_HPP_
